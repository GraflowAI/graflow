"""Docker container task execution handler."""

import base64
import logging
import pathlib
import sys
from typing import Any, Optional

from cloudpickle import __version__ as cloudpickle_version
from docker.types import DeviceRequest

import graflow
from graflow import __version__ as graflow_version
from graflow.core.context import ExecutionContext
from graflow.core.handler import TaskHandler
from graflow.core.serialization import dumps, loads
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
        image: str = "python:3.11-slim",
        auto_remove: bool = True,
        environment: Optional[dict[str, str]] = None,
        volumes: Optional[dict[str, dict[str, str]]] = None,
        device_requests: Optional[list[DeviceRequest]] = None,
        auto_mount_graflow: bool = True,
    ):
        """Initialize DockerTaskHandler.

        Args:
            image: Docker image to use
            auto_remove: Remove container after execution
            environment: Environment variables dict
            volumes: Volume mounts dict {host_path: {"bind": container_path, "mode": "rw"}}
            device_requests: GPU/device requests for docker
            auto_mount_graflow: Auto-detect and mount graflow source if running from source (default: True)
        """
        self.image = image
        self.auto_remove = auto_remove
        self.environment = environment or {}
        self.device_requests = device_requests or []
        self._warn_on_python_major_mismatch()

        # Auto-detect and mount graflow source if running from source
        self.volumes = volumes or {}
        if auto_mount_graflow:
            # Check if /graflow_src is already bound in any volume
            has_graflow_mount = any(vol_config.get("bind") == "/graflow_src" for vol_config in self.volumes.values())
            if not has_graflow_mount:
                self._auto_mount_graflow_source()

    def _auto_mount_graflow_source(self) -> None:
        """Auto-detect if graflow is running from source and mount it.

        If graflow is pip-installed, no mounting is needed (will install from PyPI).
        If graflow is running from source, mount the source directory.
        """

        graflow_path = pathlib.Path(graflow.__file__).parent.parent

        # Check if graflow is installed from source (has .git or pyproject.toml)
        is_source = (graflow_path / ".git").exists() or (graflow_path / "pyproject.toml").exists()

        if is_source:
            graflow_path_str = str(graflow_path)

            # Check if user already mounted this source path somewhere
            existing_mount = self.volumes.get(graflow_path_str)
            if existing_mount:
                # User already mounted the graflow source - use their mount
                # The template will install dependencies when it finds graflow is importable
                container_path = existing_mount.get("bind", "unknown")
                logger.info(
                    f"[DockerTaskHandler] Graflow source already mounted by user to {container_path}, "
                    "will install dependencies in container"
                )
                print(
                    f"📂 [DockerTaskHandler] Using user-provided graflow mount: {graflow_path} -> {container_path}",
                    flush=True,
                )
            else:
                # Add our own mount to /graflow_src
                logger.info(f"[DockerTaskHandler] Detected source install - mounting {graflow_path}")
                print(f"📂 [DockerTaskHandler] Mounting graflow source: {graflow_path}", flush=True)
                self.volumes[graflow_path_str] = {"bind": "/graflow_src", "mode": "ro"}
        else:
            logger.info(f"[DockerTaskHandler] Detected pip install - will install graflow=={graflow_version} from PyPI")
            print(f"📦 [DockerTaskHandler] Will install graflow=={graflow_version} from PyPI in container", flush=True)

    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
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
                "[DockerTaskHandler] docker package not installed. Install with: pip install docker"
            ) from e

        task_id = task.task_id
        logger.info(f"[DockerTaskHandler] Executing task {task_id} in Docker container")

        # Get Docker client
        client = docker.from_env()

        # Serialize task function and context
        task_code = self._serialize_task(task)
        context_code = self._serialize_context(context)

        # Load runner script from template and render with Jinja2
        runner_script = self._render_runner_script(task_code=task_code, context_code=context_code, task_id=task_id)

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

        # Stream logs in real-time, capturing CONTEXT: and ERROR: lines
        context_line, error_line = self._stream_container_logs(container)

        # Wait for container to finish
        exit_status = container.wait()

        # Parse context from captured line
        updated_context = self._parse_context_line(context_line)

        # Update result from container context when available
        result: Any = None
        if updated_context:
            result = updated_context.get_result(task_id)

        # Clean up
        if self.auto_remove:
            container.remove()

        # Check exit status
        if exit_status["StatusCode"] != 0:
            # Use captured error line or fallback to unknown error
            if error_line:
                error_msg = error_line[6:]  # Remove "ERROR:" prefix
            else:
                error_msg = "Unknown error (no ERROR: line in logs)"
            logger.error(f"[DockerTaskHandler] Container failed: {error_msg}")
            runtime_error = RuntimeError(
                f"[DockerTaskHandler] Container exited with code {exit_status['StatusCode']}: {error_msg}"
            )
            context.set_result(task_id, runtime_error)
            raise runtime_error

        # Success path: store result (may be None) for downstream tasks
        context.set_result(task_id, result)
        logger.info(f"[DockerTaskHandler] Task {task_id} completed successfully")
        return result

    def _warn_on_python_major_mismatch(self) -> None:
        """Warn when the container image Python major version differs from the host."""
        if not self.image.startswith("python:"):
            return

        tag = self.image[len("python:") :]
        if not tag:
            return

        version_part = tag.split("-", 1)[0]
        version_parts = version_part.split(".", 2)
        major_str = version_parts[0]
        minor_str = version_parts[1] if len(version_parts) > 1 else None
        if not major_str.isdigit() or (minor_str is not None and not minor_str.isdigit()):
            return

        image_major = int(major_str)
        image_minor = int(minor_str) if minor_str is not None else None
        host_major = sys.version_info.major
        host_minor = sys.version_info.minor
        if image_major != host_major or (image_minor is not None and image_minor != host_minor):
            logger.warning(
                "[DockerTaskHandler] Python version mismatch (host=%s.%s, image=%s). "
                "Unpickling may fail across Python versions.",
                host_major,
                host_minor,
                self.image,
            )

    def _serialize_task(self, task: Executable) -> str:
        """Serialize entire task object for Docker execution using cloudpickle.

        Args:
            task: Task to serialize

        Returns:
            Base64-encoded cloudpickle'd task object

        Note:
            Serializes the entire task object (not just the function) to preserve
            metadata like inject_context, inject_llm_client, etc.
        """
        # Serialize entire task object to preserve metadata
        pickled = dumps(task)
        encoded = base64.b64encode(pickled).decode("utf-8")

        return encoded

    def _serialize_context(self, context: ExecutionContext) -> str:
        """Serialize execution context for Docker execution using cloudpickle.

        Args:
            context: Execution context to serialize

        Returns:
            Base64-encoded cloudpickle'd context

        Note:
            Uses cloudpickle for better support of lambdas and closures.
            ExecutionContext.__getstate__ is automatically called.
        """
        pickled = dumps(context)
        encoded = base64.b64encode(pickled).decode("utf-8")
        return encoded

    def _stream_container_logs(self, container: Any) -> tuple[Optional[str], Optional[str]]:
        """Stream container logs in real-time and capture special lines.

        Args:
            container: Docker container instance

        Returns:
            Tuple of (context_line, error_line) - captured CONTEXT: and ERROR: lines

        Note:
            - Streams logs in real-time for user visibility
            - Captures CONTEXT: line for result deserialization
            - Captures ERROR: line for error reporting
            - Does not buffer all logs to avoid OOM
        """
        context_line = None
        error_line = None

        try:
            # Stream logs from container (combines stdout and stderr)
            for log_line in container.logs(stream=True, follow=True):
                line = log_line.decode("utf-8").rstrip()

                # Capture CONTEXT: line for parsing (but don't print it)
                if line.startswith("CONTEXT:"):
                    context_line = line
                    logger.debug(f"[DockerTaskHandler] Captured context line (length: {len(line)})")
                # Capture ERROR: line for error reporting
                elif line.startswith("ERROR:"):
                    error_line = line
                    logger.debug(f"[DockerTaskHandler] Captured error: {line}")
                # Print all other lines in real-time (stderr: status + pip output, stdout: task output)
                elif line:
                    print(line, flush=True)
        except Exception as e:
            logger.warning(f"[DockerTaskHandler] Error streaming logs: {e}")
            # Fall back to scanning logs without buffering all
            try:
                for log_line in container.logs().decode("utf-8").split("\n"):
                    if log_line.startswith("CONTEXT:"):
                        context_line = log_line
                    elif log_line.startswith("ERROR:"):
                        error_line = log_line
            except Exception as fallback_error:
                logger.error(f"[DockerTaskHandler] Failed to retrieve logs: {fallback_error}")

        return context_line, error_line

    def _parse_context_line(self, context_line: Optional[str]) -> Optional[ExecutionContext]:
        """Parse execution context from captured CONTEXT: line.

        Args:
            context_line: Line from container logs starting with "CONTEXT:"

        Returns:
            Deserialized ExecutionContext or None if parsing fails

        Note:
            Uses cloudpickle for deserialization.
            Only logs are debug-level to avoid OOM with large contexts.
        """
        if not context_line:
            logger.warning("[DockerTaskHandler] No context line captured from container")
            return None

        context_str = context_line[8:]  # Remove "CONTEXT:" prefix
        try:
            context_data = base64.b64decode(context_str)
            context = loads(context_data)
            logger.debug("[DockerTaskHandler] Successfully deserialized context from container")
            return context
        except Exception as e:
            logger.error(f"[DockerTaskHandler] Failed to deserialize context: {e}")
            return None

    def _render_runner_script(self, task_code: str, context_code: str, task_id: str) -> str:
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
                "[DockerTaskHandler] jinja2 package not installed. Install with: pip install jinja2"
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
            task_code=task_code,
            context_code=context_code,
            task_id=task_id,
            graflow_version=graflow_version,
            cloudpickle_version=cloudpickle_version,
        )
