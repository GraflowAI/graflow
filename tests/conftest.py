"""Pytest configuration and fixtures for coordination tests."""

import time

import pytest
import redis

from graflow.core.workflow import clear_workflow_context

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DOCKER_AVAILABLE = False


@pytest.fixture(scope="session")
def redis_server():
    """Real Redis server - uses local Redis if available, otherwise auto-starts Docker container."""
    # First try to connect to existing local Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        # Local Redis is available, use it
        yield redis_client
        return
    except redis.ConnectionError:
        # Local Redis not available, will try Docker
        pass

    # Try to auto-start Docker container
    container = None
    try:
        if not DOCKER_AVAILABLE:
            pytest.skip(
                "Docker package not installed. Install with: pip install docker"
            )

        assert docker is not None

        # Create Docker client
        try:
            client = docker.from_env()
            client.ping()
        except Exception as docker_err:
            # Docker daemon not running
            pytest.skip(
                f"Docker daemon not running. Please start Docker Desktop.\n"
                f"Error: {docker_err}"
            )

        # Start Redis container
        container = client.containers.run(
            "redis:7.2",
            ports={'6379/tcp': 6379},
            detach=True,
            remove=True
        )
        time.sleep(2)  # Wait for Redis to be ready

        # Create Redis client and verify connection
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        try:
            redis_client.ping()
        except redis.ConnectionError:
            time.sleep(2)  # Additional wait if needed
            redis_client.ping()

        yield redis_client

        # Cleanup container
        if container:
            container.stop()

    except Exception as e:
        # Cleanup container if it exists
        if container:
            try:
                container.stop()
            except Exception:
                pass

        # Final fallback - can't set up Redis
        if "Failed:" in str(e):
            # Already a pytest.fail, re-raise
            raise
        pytest.skip(f"Failed to start Redis container: {e}")


@pytest.fixture
def clean_redis(redis_server):
    """Provide a clean Redis instance for each test."""
    # Clean up before test
    redis_server.flushdb()
    yield redis_server
    # Clean up after test
    redis_server.flushdb()

@pytest.fixture(autouse=True)
def reset_global_workflow_context():
    """Ensure workflow ContextVar does not leak state across tests."""
    clear_workflow_context()
    yield
    clear_workflow_context()
