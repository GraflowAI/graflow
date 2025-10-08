"""
Hello World Example - The Simplest Graflow Workflow

This example demonstrates:
- How to define a task using the @task decorator
- How to execute a task
- Basic task execution model

Expected Output:
    Hello, Graflow!
    Task completed with result: success
"""

from graflow.core.decorators import task


@task
def hello():
    """A simple task that prints a greeting."""
    print("Hello, Graflow!")
    return "success"


if __name__ == "__main__":
    # Execute the task
    result = hello.run()
    print(f"Task completed with result: {result}")
