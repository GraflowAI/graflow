"""
Task with Parameters Example - Flexible Task Execution

This example demonstrates:
- Defining tasks with parameters
- Calling tasks with different arguments
- Type hints for better code clarity
- Chaining task results

Expected Output:
    Result of add(5, 3): 8
    Result of multiply(4, 7): 28
    Calculating: 5 + 3 = 8
    Then: 8 * 2 = 16
    Final formatted result: Result: 16
"""

from graflow.core.decorators import task


@task
def add(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    print(f"Calculating: {a} + {b} = {result}")
    return result


@task
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    result = x * y
    print(f"Then: {x} * {y} = {result}")
    return result


@task
def format_result(value: int) -> str:
    """Format a numeric result as a string."""
    formatted = f"Result: {value}"
    print(f"Final formatted result: {formatted}")
    return formatted


def main():
    print("=== Task with Parameters Example ===\n")

    # Example 1: Call tasks with direct parameters
    print("Example 1: Direct task calls")
    result1 = add(5, 3)
    print(f"Result of add(5, 3): {result1}\n")

    result2 = multiply(4, 7)
    print(f"Result of multiply(4, 7): {result2}\n")

    # Example 2: Chain tasks together
    print("Example 2: Chaining tasks")
    step1 = add(5, 3)
    step2 = multiply(step1, 2)
    final = format_result(step2)
    print(f"Chained result: {final}\n")

    # Example 3: Nested execution
    print("Example 3: Nested execution")
    nested_result = format_result(multiply(add(10, 5), 3))
    print(f"Nested execution result: {nested_result}")


if __name__ == "__main__":
    main()
