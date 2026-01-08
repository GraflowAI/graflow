"""Docker task runner script - Self-contained execution environment.

This script runs inside a Docker container to execute a serialized task.
It has NO dependencies on graflow being installed in the container.

Variables are substituted by Jinja2 template engine from the host.
"""

import base64
import os
import subprocess
import sys


def ensure_cloudpickle():
    """Ensure cloudpickle is available (auto-install if missing)."""
    cloudpickle_version = "{{ cloudpickle_version }}"
    try:
        import cloudpickle

        return cloudpickle
    except ImportError:
        # Cloudpickle not installed - install it automatically
        print(f"\n[DockerTaskRunner] Installing cloudpickle=={cloudpickle_version}", file=sys.stderr, flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", f"cloudpickle=={cloudpickle_version}"],
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            import cloudpickle

            print(
                f"[DockerTaskRunner] ✅ cloudpickle=={cloudpickle_version} installed successfully\n",
                file=sys.stderr,
                flush=True,
            )
            return cloudpickle
        except Exception as e:
            print(
                f"[DockerTaskRunner] ❌ Failed to install cloudpickle=={cloudpickle_version}: {e}",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)


def ensure_graflow():
    """Ensure graflow is available (required for ExecutionContext deserialization).

    Strategy:
    1. If import graflow works → use it directly
    2. If /graflow_src is mounted → install from there
    3. Otherwise → install from PyPI
    """

    # Step 1: Try to import graflow
    try:
        import graflow

        return graflow
    except ImportError:
        pass

    if os.path.exists("/graflow_src"):
        # Step 2-a: Install from /graflow_src if mounted
        print("[DockerTaskRunner] Installing graflow from mounted source...", file=sys.stderr, flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "/graflow_src"],
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            import graflow

            print("[DockerTaskRunner] ✅ graflow installed from mounted source\n", file=sys.stderr, flush=True)
            return graflow
        except Exception as e:
            import traceback

            print(f"[DockerTaskRunner] ❌ Failed to install from /graflow_src: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        # Step 2-b: Install from PyPI
        graflow_version = "{{ graflow_version }}"
        print(f"[DockerTaskRunner] Installing graflow=={graflow_version} from PyPI...", file=sys.stderr, flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", f"graflow=={graflow_version}"],
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            import graflow

            print(
                f"[DockerTaskRunner] ✅ graflow=={graflow_version} installed from PyPI\n", file=sys.stderr, flush=True
            )
            return graflow
        except Exception as e:
            print(f"[DockerTaskRunner] ❌ Failed to install graflow: {e}", file=sys.stderr, flush=True)
            print(
                "[DockerTaskRunner] Hint: Mount graflow source to /graflow_src in container",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)


# Step 1: Ensure dependencies are available
cloudpickle = ensure_cloudpickle()
graflow = ensure_graflow()


# Step 2: Define serialization functions (bundled from graflow.core.serialization)
def dumps(obj):
    """Serialize object using cloudpickle."""
    return cloudpickle.dumps(obj)


def loads(data):
    """Deserialize object using cloudpickle."""
    return cloudpickle.loads(data)


# Step 3: Execute task in isolated environment
context = None
task_id = "{{ task_id }}"

try:
    # Deserialize task object and execution context
    task_data = base64.b64decode("{{ task_code }}")
    task = loads(task_data)

    context_data = base64.b64decode("{{ context_code }}")
    context = loads(context_data)

    # Set execution context on the task
    # This is needed for context injection and task execution
    task.set_execution_context(context)

    # Execute task using its run() method
    # This properly handles inject_context, inject_llm_client, etc.
    with context.executing_task(task):
        result = task.run()

    # Store result in context (inside container)
    context.set_result(task_id, result)

    # Serialize updated context for return to host
    updated_context = dumps(context)
    encoded_context = base64.b64encode(updated_context).decode("utf-8")
    print(f"CONTEXT:{encoded_context}")

except Exception as e:
    # Handle execution errors
    if context is not None:
        # Store exception in context if context was successfully deserialized
        context.set_result(task_id, e)

        # Serialize updated context with error
        updated_context = dumps(context)
        encoded_context = base64.b64encode(updated_context).decode("utf-8")
        print(f"CONTEXT:{encoded_context}")

    # Print error for host to parse
    print(f"ERROR:{str(e)}", file=sys.stderr)
    sys.exit(1)
