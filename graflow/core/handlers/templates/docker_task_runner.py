"""Docker task runner script.

This script runs inside a Docker container to execute a task.
Variables are substituted by Jinja2 template engine.
"""
import base64
import pickle
import sys

# Initialize variables to None for exception handling
context = None
task_id = '{{ task_id }}'

try:
    # Deserialize task and context
    task_data = base64.b64decode('{{ task_code }}')
    task_func = pickle.loads(task_data)

    context_data = base64.b64decode('{{ context_code }}')
    context = pickle.loads(context_data)

    # Execute task
    result = task_func()

    # Store result in context (inside container)
    context.set_result(task_id, result)

    # Serialize updated context for return
    updated_context = pickle.dumps(context)
    encoded_context = base64.b64encode(updated_context).decode('utf-8')
    print(f"CONTEXT:{encoded_context}")

except Exception as e:
    # Store exception in context if context was successfully deserialized
    if context is not None:
        context.set_result(task_id, e)

        # Serialize updated context with error
        updated_context = pickle.dumps(context)
        encoded_context = base64.b64encode(updated_context).decode('utf-8')
        print(f"CONTEXT:{encoded_context}")

    print(f"ERROR:{str(e)}", file=sys.stderr)
    sys.exit(1)
