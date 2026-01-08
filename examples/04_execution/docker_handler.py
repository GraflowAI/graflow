"""
Docker Handler Example
=======================

This example demonstrates the DockerTaskHandler - executing tasks inside
isolated Docker containers. This provides process isolation and environment
control, useful for untrusted code, specific dependencies, or reproducible
environments.

⚠️  Prerequisites:
-----------------
- Docker installed and running
- Docker SDK for Python: pip install docker
- Python image available: python:3.11-slim

Note: Both cloudpickle and graflow are automatically installed in the container.
      - If running from source: graflow source is auto-mounted and installed
      - If pip-installed: graflow is installed from PyPI (version-matched)
      For production, consider using a custom image with dependencies pre-installed.

To check if Docker is available:
    docker --version
    docker ps

Concepts Covered:
-----------------
1. Docker handler specification
2. Container execution and isolation
3. ExecutionContext serialization and deserialization
4. Context injection in Docker tasks (inject_context=True)
5. Result retrieval from containers
6. Performance overhead comparison
7. When to use Docker vs Direct handler

Expected Output:
----------------
=== Docker Handler Demo ===

Checking Docker availability...
✅ Docker is available

📂 [DockerTaskHandler] Mounting graflow source: /path/to/graflow
✅ Local Task (Direct Handler)
   Executing in main process
   Python version: 3.11.x
   Process ID: 12345

[DockerTaskRunner] Installing cloudpickle...
[DockerTaskRunner] ✅ cloudpickle installed successfully
[DockerTaskRunner] Installing graflow from mounted source...
[DockerTaskRunner] ✅ graflow installed successfully from mounted source
🐳 Docker Task (Docker Handler)
   Executing in Docker container
   Python version: 3.11.x
   Container process ID: 1
   Previous task result (via context): local_result

✅ Compare Task
   Local task result: local_result
   Docker task result: docker_result
   Both handlers successfully executed!

=== Summary ===
✅ DirectTaskHandler: Fast, in-process execution
🐳 DockerTaskHandler: Isolated container execution
   - First run: ~3-5s overhead (auto-installs cloudpickle + graflow)
   - Subsequent runs: ~500-1000ms overhead (container startup only)
   - Full process isolation
   - ExecutionContext works across container boundaries
   - Use when isolation is needed
"""

import os
import sys


def check_docker_available():
    """Check if Docker is available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def main():
    """Demonstrate Docker handler usage."""
    print("=== Docker Handler Demo ===\n")

    # Check Docker availability
    print("Checking Docker availability...")
    if not check_docker_available():
        print("❌ Docker is not available.")
        print("\nTo run this example, you need:")
        print("1. Docker installed and running")
        print("2. Docker SDK: pip install docker")
        print("3. Python image: docker pull python:3.11-slim")
        print("\nSkipping Docker example...")
        return

    print("✅ Docker is available\n")

    # Import after checking Docker is available
    from graflow.core.context import TaskExecutionContext
    from graflow.core.decorators import task
    from graflow.core.engine import WorkflowEngine
    from graflow.core.handlers.docker import DockerTaskHandler
    from graflow.core.workflow import workflow

    with workflow("docker_demo") as ctx:

        @task(handler="direct")
        def task_local():
            """Execute locally with DirectTaskHandler."""
            print("✅ Local Task (Direct Handler)")
            print("   Executing in main process")
            print(f"   Python version: {sys.version.split()[0]}")
            print(f"   Process ID: {os.getpid()}\n")
            return "local_result"

        @task(handler="docker", inject_context=True)
        def task_docker(context: TaskExecutionContext):
            """
            Execute in Docker container with DockerTaskHandler.

            This task runs in complete isolation from the main process.
            It demonstrates that ExecutionContext is properly serialized,
            deserialized, and functional inside the container.
            """
            import os
            import sys

            print("🐳 Docker Task (Docker Handler)")
            print("   Executing in Docker container")
            print(f"   Python version: {sys.version.split()[0]}")
            print(f"   Container process ID: {os.getpid()}")

            # Demonstrate context access works in Docker
            local_result = context.get_result("task_local")
            print(f"   Previous task result (via context): {local_result}")
            print()

            return "docker_result"

        @task(handler="direct", inject_context=True)
        def compare_results(context: TaskExecutionContext):
            """Compare results from both handlers."""

            print("✅ Compare Task")

            # Get results from previous tasks
            local_result = context.get_result("task_local")
            docker_result = context.get_result("task_docker")

            print(f"   Local task result: {local_result}")
            print(f"   Docker task result: {docker_result}")
            print("   Both handlers successfully executed!\n")

            return {"local": local_result, "docker": docker_result}

        # Define workflow
        _ = task_local >> task_docker >> compare_results

        # Register Docker handler with the engine
        engine = WorkflowEngine()

        # DockerTaskHandler auto-detects if graflow is running from source
        # and mounts it automatically. If graflow is pip-installed, it will
        # install the same version from PyPI in the container.
        engine.register_handler("docker", DockerTaskHandler(image="python:3.11-slim"))

        # Create execution context
        from graflow.core.context import ExecutionContext

        exec_context = ExecutionContext.create(ctx.graph, "task_local", max_steps=10)

        # Execute with custom engine
        engine.execute(exec_context)

    # Summary
    print("=== Summary ===")
    print("✅ DirectTaskHandler: Fast, in-process execution")
    print("🐳 DockerTaskHandler: Isolated container execution")
    print("   - First run: ~3-5s overhead (auto-installs cloudpickle + graflow)")
    print("   - Subsequent runs: ~500-1000ms overhead (container startup only)")
    print("   - Full process isolation")
    print("   - ExecutionContext works across container boundaries")
    print("   - Use when isolation is needed")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Docker Handler Specification**
#    @task(handler="docker")
#    def my_task():
#        pass
#
#    - Specify handler="docker" in task decorator
#    - Task will execute in isolated Docker container
#    - Requires Docker to be installed and running
#
# 2. **Container Isolation**
#    - Separate process space
#    - Separate filesystem
#    - Separate network (by default)
#    - Cannot directly access host resources
#
# 3. **Serialization**
#    - Task function is serialized (cloudpickle)
#    - ExecutionContext is serialized
#    - Results are deserialized back
#    - Works with lambdas and closures
#
# 4. **Performance Overhead**
#    - Container startup: ~500-2000ms
#    - Serialization/deserialization: ~10-100ms
#    - Execution: Similar to direct (native Python)
#    - Total overhead: Significant compared to DirectTaskHandler
#
# 5. **When to Use Docker Handler**
#    ✅ Untrusted code (security isolation)
#    ✅ Different Python versions
#    ✅ Specific system dependencies
#    ✅ Reproducible environments
#    ✅ Testing in clean environment
#    ✅ Resource limits (CPU, memory)
#
# 6. **When NOT to Use Docker Handler**
#    ❌ Performance-critical code
#    ❌ Frequent short tasks (overhead dominates)
#    ❌ Tasks needing host filesystem access
#    ❌ Development iteration (slow feedback)
#    ❌ Most production tasks (use direct)
#
# 7. **Handler Registration**
#    engine = WorkflowEngine()
#    engine.register_handler("docker", DockerTaskHandler(
#        image="python:3.11-slim",
#        environment={"API_KEY": "secret"},
#        volumes={"/host": {"bind": "/container", "mode": "ro"}}
#    ))
#
#    - Register handler before executing workflow
#    - Configure image, volumes, environment
#    - Can register multiple docker handlers with different configs
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Use different Docker images:
#    @task(handler="docker")
#    def numpy_task():
#        import numpy as np
#        return np.array([1, 2, 3]).sum()
#
#    # Register with numpy image
#    engine.register_handler("docker", DockerTaskHandler(
#        image="python:3.11-slim"
#    ))
#    # Note: Would need to install numpy in container or use image with numpy
#
# 2. Pass environment variables:
#    engine.register_handler("docker", DockerTaskHandler(
#        image="python:3.11-slim",
#        environment={"DEBUG": "1", "API_URL": "https://api.example.com"}
#    ))
#
#    @task(handler="docker")
#    def use_env():
#        import os
#        return os.getenv("DEBUG")
#
# 3. Mount volumes:
#    engine.register_handler("docker", DockerTaskHandler(
#        image="python:3.11-slim",
#        volumes={
#            "/path/on/host": {
#                "bind": "/data",
#                "mode": "ro"  # read-only
#            }
#        }
#    ))
#
#    @task(handler="docker")
#    def read_file():
#        with open("/data/file.txt") as f:
#            return f.read()
#
# 4. Compare execution times:
#    import time
#
#    @task(handler="direct")
#    def fast():
#        start = time.time()
#        # work
#        return time.time() - start
#
#    @task(handler="docker")
#    def slow():
#        import time
#        start = time.time()
#        # same work
#        return time.time() - start
#
# 5. Test with untrusted code:
#    @task(handler="docker")
#    def untrusted():
#        # This runs isolated - can't harm host
#        import os
#        files = os.listdir("/")  # Container's root, not host
#        return files
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **CI/CD Pipeline**:
# Run tests in isolated containers to ensure reproducibility
#
# **Multi-Version Testing**:
# Test code against different Python versions
# handler_py39 = DockerTaskHandler(image="python:3.9")
# handler_py311 = DockerTaskHandler(image="python:3.11")
#
# **Data Science Experiments**:
# Run experiments with specific dependency versions
# handler = DockerTaskHandler(image="tensorflow/tensorflow:latest-gpu")
#
# **Security Sandbox**:
# Execute user-provided code safely in isolated container
#
# **Dependency Isolation**:
# Task requires specific library versions different from main env
#
# ============================================================================
# Advanced Configuration:
# ============================================================================
#
# **GPU Support**:
# from docker.types import DeviceRequest
#
# handler = DockerTaskHandler(
#     image="tensorflow/tensorflow:latest-gpu",
#     device_requests=[
#         DeviceRequest(count=1, capabilities=[["gpu"]])
#     ]
# )
#
# **Resource Limits**:
# # Note: Requires direct docker API usage
# container = client.containers.run(
#     image="python:3.11",
#     mem_limit="512m",  # 512MB RAM
#     cpu_period=100000,
#     cpu_quota=50000,  # 50% CPU
# )
#
# **Network Configuration**:
# handler = DockerTaskHandler(
#     image="python:3.11",
#     # network_mode="bridge"  # Default
#     # network_mode="host"  # Share host network
#     # network_mode="none"  # No network
# )
#
# ============================================================================
# Troubleshooting:
# ============================================================================
#
# **Docker not running**:
# Error: Cannot connect to Docker daemon
# Solution: Start Docker Desktop or Docker service
#   - macOS: Open Docker Desktop
#   - Linux: sudo systemctl start docker
#   - Windows: Start Docker Desktop
#
# **Image not found**:
# Error: Image python:3.11 not found
# Solution: Pull the image first
#   docker pull python:3.11-slim
#
# **Permission denied**:
# Error: Permission denied while trying to connect to Docker daemon
# Solution: Add user to docker group (Linux)
#   sudo usermod -aG docker $USER
#   # Then log out and back in
#
# **Serialization errors**:
# Error: cannot pickle 'module' object
# Solution: Don't import modules at module level, import inside task
#
# **Volume mount errors**:
# Error: invalid mount config
# Solution: Use absolute paths for host directories
#   volumes={"/absolute/path": {"bind": "/container", "mode": "rw"}}
#
# ============================================================================
