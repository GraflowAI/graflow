"""
Multi-Image Docker Pipeline
============================

Demonstrates running different tasks of the same workflow in **different
Docker images** by registering multiple ``DockerTaskHandler`` instances
under distinct handler names and selecting them per-task via
``@task(handler="<name>")``.

This is the advanced counterpart to ``examples/04_execution/docker_handler.py``
— instead of using a single Docker image, we mix:

- A local host step (no container overhead, fast access to local state)
- An ETL step running inside a slim Python image
- A training-style step running inside a heavier image with extra deps

The same workflow code ``local >> etl >> train`` transparently routes each
task to the right runtime.

⚠️  Prerequisites:
-----------------
- Docker installed and running
- Docker SDK for Python: ``pip install docker``
- Images pulled (first run will pull automatically):
    - ``python:3.11-slim``
    - ``python:3.11`` (stands in for a heavier / specialized image)

⚠️  Python Version Must Match Across Containers:
-----------------------------------------------
Tasks are shipped between host and containers via ``cloudpickle``, whose
serialized bytecode is Python-version-specific. When you chain multiple
Docker tasks, **all containers (and the host) must run the same Python
minor version** (e.g., all 3.11). Mixing 3.11 and 3.12 will fail with
errors like ``module 'sys' has no attribute 'sys'`` during deserialization.

The per-task handler mechanism is still fully useful for mixing images
with *different dependencies, sizes, or base layers* — e.g.,
``python:3.11-slim`` for ETL vs. ``pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime``
(which ships Python 3.11) for training.

Concepts Covered:
-----------------
1. Registering multiple DockerTaskHandler instances under distinct names
2. Selecting a handler per task with ``@task(handler="<name>")``
3. Mixing host-local and containerized execution in one workflow
4. Accessing upstream task results across container boundaries

Expected Output:
----------------
=== Multi-Image Docker Pipeline ===

Checking Docker availability...
✅ Docker is available

[host]         local_step   -> python 3.11.x (pid=...)
[py311-slim]   etl_step     -> python 3.11.x (pid=1)
[py311-full]   train_step   -> python 3.11.x (pid=1)

=== Final Results ===
  local_step: ran on host (python 3.11.x)
  etl_step:   etl on python 3.11.x
  train_step: trained on python 3.11.x (upstream: 'etl on python 3.11.x')
"""

def check_docker_available() -> bool:
    """Return True if a Docker daemon is reachable.

    Tries DOCKER_HOST/default first, then common macOS fallbacks
    (Docker Desktop, colima) since ``docker.from_env()`` does not resolve
    the CLI's context file.
    """
    import os as _os
    import pathlib as _pathlib
    from typing import Optional

    import docker

    candidates: list[Optional[str]] = [None]  # docker.from_env() (DOCKER_HOST or /var/run/docker.sock)
    home = _pathlib.Path.home()
    for sock in (home / ".docker/run/docker.sock", home / ".colima/default/docker.sock"):
        if sock.exists():
            candidates.append(f"unix://{sock}")

    last_err: Exception | None = None
    for base_url in candidates:
        try:
            client = docker.DockerClient(base_url=base_url) if base_url else docker.from_env()
            client.ping()
            if base_url:
                # Propagate so DockerTaskHandler / SDK-using code picks the same socket
                _os.environ["DOCKER_HOST"] = base_url
            return True
        except Exception as err:
            last_err = err

    if last_err is not None:
        print(f"   (last error: {last_err})")
    return False


def main() -> None:
    print("=== Multi-Image Docker Pipeline ===\n")

    print("Checking Docker availability...")
    if not check_docker_available():
        print("❌ Docker is not available. Install Docker + `pip install docker` and retry.")
        return
    print("✅ Docker is available\n")

    # Deferred imports so the example can be collected even without Docker installed
    from graflow import task, workflow
    from graflow.core.context import ExecutionContext
    from graflow.core.engine import WorkflowEngine
    from graflow.core.handlers.docker import DockerTaskHandler

    with workflow("multi_image_pipeline") as wf:

        @task
        def local_step() -> str:
            """Runs on the host process — no container."""
            import os
            import sys

            py = sys.version.split()[0]
            print(f"[host]       local_step   -> python {py} (pid={os.getpid()})")
            return f"ran on host (python {py})"

        @task(handler="py311-slim")
        def etl_step() -> str:
            """Runs inside the ``python:3.11-slim`` container."""
            import os
            import sys

            py = sys.version.split()[0]
            print(f"[py311-slim] etl_step     -> python {py} (pid={os.getpid()})")
            return f"etl on python {py}"

        @task(handler="py311-full", inject_context=True)
        def train_step(context):
            """Runs inside a different image; reads upstream results via context."""
            import os
            import sys

            py = sys.version.split()[0]
            upstream = context.get_result("etl_step")
            print(f"[py311-full] train_step   -> python {py} (pid={os.getpid()})")
            print(f"             upstream etl: {upstream!r}")
            return f"trained on python {py} (upstream: {upstream!r})"

        local_step >> etl_step >> train_step  # type: ignore[operator]

        # Each registered name binds to exactly one image.
        # A task's handler="<name>" picks which runtime to use.
        engine = WorkflowEngine()
        engine.register_handler("py311-slim", DockerTaskHandler(image="python:3.11-slim"))
        engine.register_handler("py311-full", DockerTaskHandler(image="python:3.11"))

        exec_context = ExecutionContext.create(wf.graph, "local_step", max_steps=10)
        engine.execute(exec_context)

    print("\n=== Final Results ===")
    for name in ("local_step", "etl_step", "train_step"):
        print(f"  {name}: {exec_context.get_result(name)}")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways
# ============================================================================
#
# 1. **One handler name == one image.**
#    ``DockerTaskHandler`` fixes its image at construction. To run tasks
#    in different images, register multiple handlers under distinct names.
#
# 2. **`@task(handler="<name>")` is the router.**
#    The decorator picks which registered handler runs the task.
#    Tasks without a handler argument use the default (host/direct) handler.
#
# 3. **Context crosses container boundaries.**
#    ``context.get_result(...)`` and channel data are serialized in/out of
#    the container. Downstream tasks can always read upstream results.
#
# 4. **Typical real-world setups.**
#    - CPU preprocessing in ``python:3.11-slim``
#    - GPU training in ``pytorch/pytorch:latest``
#    - Legacy code in ``python:3.9``
#    - Vendor-specific SDKs in custom images
#    All composed with plain ``a >> b >> c``.
#
# ============================================================================
# Try Experimenting
# ============================================================================
#
# 1. Swap one of the images for a GPU-enabled runtime:
#       engine.register_handler(
#           "gpu_train",
#           DockerTaskHandler(image="pytorch/pytorch:latest"),
#       )
#    and tag a task with @task(handler="gpu_train").
#
# 2. Use a custom image built from your own Dockerfile with the dependencies
#    pre-installed — first-run overhead drops to plain container startup.
#
# 3. Fan out to multiple containers in parallel:
#       local_step >> (etl_a | etl_b | etl_c) >> train_step
# ============================================================================
