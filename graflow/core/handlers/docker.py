"""Docker container task execution handler."""

import base64
import logging
import pickle
from typing import Optional

from docker.types import DeviceRequest

from graflow.core.context import ExecutionContext
from graflow.core.handler import TaskHandler
from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class DockerTaskHandler(TaskHandler):
    """Execute tasks inside Docker containers.

    This handler runs tasks in isolated Docker containers, providing
    process isolation and environment control. Useful for:
    - GPU-accelerated tasks
    - Tasks requiring specific system dependencies
    - Isolated execution environments

    Attributes:
        image: Docker image name to use for execution
        auto_remove: Whether to automatically remove containers after execution
        environment: Additional environment variables to pass to container
        volumes: Volume mounts for the container

    Examples:
        >>> handler = DockerTaskHandler(image="python:3.11")
        >>> handler.execute_task(my_task, context)

        >>> # With GPU support
        >>> handler = DockerTaskHandler(
        ...     image="pytorch/pytorch:latest",
        ...     device_requests=[{"count": 1, "capabilities": [["gpu"]]}]
        ... )
    """

    def __init__(
        self,
        image: str = "python:3.11",
        auto_remove: bool = True,
        environment: Optional[dict[str, str]] = None,
        volumes: Optional[dict[str, dict[str, str]]] = None,
        device_requests: Optional[list[DeviceRequest]] = None,
    ):
        """Initialize DockerTaskHandler.

        Args:
            image: Docker image to use
            auto_remove: Remove container after execution
            environment: Environment variables dict
            volumes: Volume mounts dict {host_path: {"bind": container_path, "mode": "rw"}}
            device_requests: GPU/device requests for docker
        """
        self.image = image
        self.auto_remove = auto_remove
        self.environment = environment or {}
        self.volumes = volumes or {}
        self.device_requests = device_requests or []

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task inside a Docker container and store result in context.

        Args:
            task: Executable task to run
            context: Execution context

        Note:
            The context is serialized and passed to the container,
            where context.set_result() is called inside the container.
        """
        try:
            import docker
        except ImportError as e:
            raise ImportError(
                "[DockerTaskHandler] docker package not installed. "
                "Install with: pip install docker"
            ) from e

        task_id = task.task_id
        logger.info(f"[DockerTaskHandler] Executing task {task_id} in Docker container")

        # Get Docker client
        client = docker.from_env()

        # Serialize task function and context
        task_code = self._serialize_task(task)
        context_code = self._serialize_context(context)

        # Load runner script from template and render with Jinja2
        runner_script = self._render_runner_script(
            task_code=task_code, context_code=context_code, task_id=task_id
        )

        # Run container
        logger.debug(f"[DockerTaskHandler] Running container with image: {self.image}")

        container = client.containers.run(
            self.image,
            command=["python", "-c", runner_script],
            environment=self.environment,
            volumes=self.volumes,
            device_requests=self.device_requests if self.device_requests else None,
            detach=True,
            auto_remove=False,  # We'll remove manually after getting logs
        )

        # Wait for container to finish
        exit_status = container.wait()

        # Get logs
        logs = container.logs().decode("utf-8")
        logger.debug(f"[DockerTaskHandler] Container logs:\n{logs}")

        # Parse updated context from logs
        updated_context = self._parse_context_from_logs(logs)

        # Update our context with the results from container
        if updated_context:
            # Get result from container context and set it in our context
            result = updated_context.get_result(task_id)
            context.set_result(task_id, result)

        # Clean up
        if self.auto_remove:
            container.remove()

        # Check exit status
        if exit_status["StatusCode"] != 0:
            error_msg = self._parse_error_from_logs(logs)
            logger.error(f"[DockerTaskHandler] Container failed: {error_msg}")
            raise RuntimeError(
                f"[DockerTaskHandler] Container exited with code {exit_status['StatusCode']}: {error_msg}"
            )

        logger.info(f"[DockerTaskHandler] Task {task_id} completed successfully")

    def _serialize_task(self, task: Executable) -> str:
        """Serialize task function for Docker execution.

        Args:
            task: Task to serialize

        Returns:
            Base64-encoded pickled task function
        """
        # Get the task function
        if hasattr(task, "func"):
            task_func = task.func  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Task {task.task_id} does not have a callable function")

        # Pickle and encode
        pickled = pickle.dumps(task_func)
        encoded = base64.b64encode(pickled).decode("utf-8")

        return encoded

    def _serialize_context(self, context: ExecutionContext) -> str:
        """Serialize execution context for Docker execution.

        Args:
            context: Execution context to serialize

        Returns:
            Base64-encoded pickled context
        """
        pickled = pickle.dumps(context)
        encoded = base64.b64encode(pickled).decode("utf-8")
        return encoded

    def _parse_context_from_logs(self, logs: str) -> Optional[ExecutionContext]:
        """Parse updated context from container logs.

        Args:
            logs: Container logs string

        Returns:
            Deserialized ExecutionContext or None
        """
        # Look for CONTEXT: line in logs
        for line in logs.split("\n"):
            if line.startswith("CONTEXT:"):
                context_str = line[8:]  # Remove "CONTEXT:" prefix
                try:
                    context_data = base64.b64decode(context_str)
                    context = pickle.loads(context_data)
                    return context
                except Exception as e:
                    logger.error(
                        f"[DockerTaskHandler] Failed to deserialize context: {e}"
                    )
                    return None

        # No context found
        logger.warning("[DockerTaskHandler] No context found in container logs")
        return None

    def _parse_error_from_logs(self, logs: str) -> str:
        """Parse error message from container logs.

        Args:
            logs: Container logs string

        Returns:
            Error message or empty string
        """
        # Look for ERROR: line in logs
        for line in logs.split("\n"):
            if line.startswith("ERROR:"):
                return line[6:]  # Remove "ERROR:" prefix

        return "Unknown error"

    def _render_runner_script(
        self, task_code: str, context_code: str, task_id: str
    ) -> str:
        """Render runner script from Jinja2 template.

        Args:
            task_code: Base64-encoded serialized task
            context_code: Base64-encoded serialized context
            task_id: Task ID

        Returns:
            Rendered Python script to execute in container
        """
        try:
            from jinja2 import Environment, PackageLoader, select_autoescape
        except ImportError as e:
            raise ImportError(
                "[DockerTaskHandler] jinja2 package not installed. "
                "Install with: pip install jinja2"
            ) from e

        # Create Jinja2 environment
        env = Environment(
            loader=PackageLoader("graflow.core.handlers", "templates"),
            autoescape=select_autoescape(),
        )

        # Load template
        template = env.get_template("docker_task_runner.py")

        # Render with variables
        return template.render(
            task_code=task_code, context_code=context_code, task_id=task_id
        )
