"""Pytest configuration and fixtures for integration tests."""

import time

import pytest
import redis

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DOCKER_AVAILABLE = False


@pytest.fixture(scope="session")
def redis_server():
    """Real Redis server using Docker container for integration tests."""
    container = None
    try:
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker not available")
        assert docker is not None, "Docker client must be available for Redis integration tests"
        # Try to use Docker first
        client = docker.from_env()
        # Test connection
        client.ping()
        container = client.containers.run(
            "redis:7.2",
            ports={'6379/tcp': 6379},
            detach=True,
            remove=True
        )
        time.sleep(2)  # Wait for Redis to be ready

        # Create Redis client and verify connection
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        try:
            redis_client.ping()
        except redis.ConnectionError:
            time.sleep(2)  # Additional wait if needed
            redis_client.ping()

        yield redis_client

        # Cleanup
        container.stop()

    except Exception as e:
        # Fallback to local Redis if Docker is not available
        if container:
            try:
                container.stop()
            except Exception:
                pass

        # Try to connect to local Redis
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
            redis_client.ping()
            pytest.skip(f"Docker not available ({e}), skipping integration tests that require isolated Redis")
        except redis.ConnectionError:
            pytest.skip(f"Neither Docker nor local Redis available for integration tests: {e}")


@pytest.fixture
def clean_redis(redis_server):
    """Provide a clean Redis instance for each test."""
    # Clean up before test
    redis_server.flushdb()
    yield redis_server
    # Clean up after test
    redis_server.flushdb()
