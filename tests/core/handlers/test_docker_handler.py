"""Tests for DockerTaskHandler."""

import sys

from docker.types import DeviceRequest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.handlers.docker import DockerTaskHandler


def is_docker_available():
    """Check if Docker daemon is available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


DOCKER_AVAILABLE = is_docker_available()
HOST_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
PYTHON_IMAGE_SLIM = f"python:{HOST_PYTHON_VERSION}-slim"
PYTHON_IMAGE_ALPINE = f"python:{HOST_PYTHON_VERSION}-alpine"
PYTHON_IMAGE = f"python:{HOST_PYTHON_VERSION}"


class TestDockerTaskHandler:
    """Test DockerTaskHandler."""

    def test_docker_task_handler_simple(self):
        """Test simple task execution in Docker."""

        @task
        def simple_task():
            return "hello from docker"

        graph = TaskGraph()
        graph.add_node(simple_task, "simple_task")
        context = ExecutionContext.create(graph, start_node="simple_task")
        simple_task.set_execution_context(context)

        # Use auto-mount feature - graflow source will be auto-mounted to /graflow_src
        handler = DockerTaskHandler(image=PYTHON_IMAGE_SLIM)
        result = handler.execute_task(simple_task, context)

        assert context.get_result("simple_task") == "hello from docker"
        assert result == "hello from docker"

    def test_docker_task_handler_with_computation(self):
        """Test task with computation."""

        @task
        def calc_task():
            return sum(range(100))

        graph = TaskGraph()
        graph.add_node(calc_task, "calc_task")
        context = ExecutionContext.create(graph, start_node="calc_task")
        calc_task.set_execution_context(context)

        # Use auto-mount feature - graflow source will be auto-mounted to /graflow_src
        handler = DockerTaskHandler(image=PYTHON_IMAGE_SLIM)
        result = handler.execute_task(calc_task, context)

        assert context.get_result("calc_task") == 4950
        assert result == 4950

    def test_docker_task_handler_custom_image(self):
        """Test with custom Docker image."""
        handler = DockerTaskHandler(image=PYTHON_IMAGE_ALPINE, auto_remove=True)
        assert handler.image == PYTHON_IMAGE_ALPINE
        assert handler.auto_remove is True

    def test_docker_task_handler_with_environment(self):
        """Test with custom environment variables."""
        handler = DockerTaskHandler(image=PYTHON_IMAGE_SLIM, environment={"MY_VAR": "test_value"})
        assert handler.environment == {"MY_VAR": "test_value"}

    def test_docker_task_handler_initialization(self):
        """Test DockerTaskHandler initialization with various parameters."""
        device_req = DeviceRequest(count=1, capabilities=[["gpu"]])
        handler = DockerTaskHandler(
            image=PYTHON_IMAGE,
            auto_remove=False,
            environment={"KEY": "value"},
            volumes={"/host/path": {"bind": "/container/path", "mode": "rw"}},
            device_requests=[device_req],
        )

        assert handler.image == "python:3.11"
        assert handler.auto_remove is False
        assert handler.environment == {"KEY": "value"}

        # Verify user-provided volume is present
        assert "/host/path" in handler.volumes
        assert handler.volumes["/host/path"] == {"bind": "/container/path", "mode": "rw"}

        # Verify graflow source is auto-mounted (appended to user volumes)
        # Should have 2 volumes: user-provided + auto-mounted graflow
        assert len(handler.volumes) == 2

        # Find the graflow volume (the one that's not /host/path)
        graflow_volumes = [
            (path, config) for path, config in handler.volumes.items() if config.get("bind") == "/graflow_src"
        ]
        assert len(graflow_volumes) == 1, "Should have exactly one graflow source mount"

        assert handler.device_requests == [device_req]
