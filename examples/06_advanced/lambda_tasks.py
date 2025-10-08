"""
Lambda and Closure Tasks
=========================

This example demonstrates cloudpickle serialization for lambda functions and closures.

Prerequisites:
--------------
1. pip install cloudpickle (installed with graflow)

Concepts Covered:
-----------------
1. Lambda tasks
2. Closure tasks capturing outer scope
3. Cloudpickle serialization
4. Dynamic task creation
5. Factory pattern for task generation

Expected Output:
----------------
=== Lambda and Closure Tasks ===

Step 1: Creating tasks from lambdas and closures
✅ Lambda task created
✅ Closure task created
✅ Complex closure task created

Step 2: Building workflow
✅ Task graph built

Step 3: Executing workflow
⏳ Executing tasks...
✅ lambda_task executed -> 42
✅ double_task executed -> 84
✅ multiply_by_10 executed -> 840

Step 4: Results
Lambda result: 42
Double result: 84
Multiply by 10 result: 840

=== Summary ===
✅ Lambda tasks work with cloudpickle
✅ Closures captured outer scope correctly
✅ All tasks serialized and executed
"""

from graflow.core.context import ExecutionContext
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


def create_multiplier(factor: int):
    """Factory function that creates a closure.

    The returned function captures 'factor' from the outer scope,
    demonstrating cloudpickle's ability to serialize closures.
    """
    def multiply(x: int) -> int:
        return x * factor  # Captures 'factor' from outer scope
    return multiply


def create_operation(operation_name: str):
    """Factory that creates operation closures.

    Demonstrates capturing both data and functions from outer scope.
    """
    operations = {
        "double": lambda x: x * 2,
        "square": lambda x: x ** 2,
        "increment": lambda x: x + 1
    }

    operation = operations.get(operation_name)

    def apply_operation(x: int) -> int:
        if operation is None:
            raise ValueError(f"Unknown operation: {operation_name}")
        return operation(x)

    return apply_operation


def main():
    """Run lambda and closure tasks demonstration."""
    print("=== Lambda and Closure Tasks ===\n")

    # Step 1: Create tasks
    print("Step 1: Creating tasks from lambdas and closures")

    # Simple lambda task
    lambda_func = lambda: 42

    # Closure that captures a value
    double_func = create_operation("double")

    # Closure with custom factor
    multiply_by_10 = create_multiplier(10)

    print("✅ Lambda task created")
    print("✅ Closure task created")
    print("✅ Complex closure task created")

    # Step 2: Build workflow
    print("\nStep 2: Building workflow")

    graph = TaskGraph()

    # Add lambda task
    lambda_task = TaskWrapper("lambda_task", lambda_func)
    graph.add_node(lambda_task, "lambda_task")

    # Add closure task that uses previous result
    def double_wrapper(context):
        """Wrapper to get previous result and apply closure."""
        prev_result = context.get_result("lambda_task")
        return double_func(prev_result)

    double_task = TaskWrapper("double_task", double_wrapper, inject_context=True)
    graph.add_node(double_task, "double_task")

    # Add multiplication task
    def multiply_wrapper(context):
        """Wrapper to get previous result and apply multiplication closure."""
        prev_result = context.get_result("double_task")
        return multiply_by_10(prev_result)

    multiply_task = TaskWrapper("multiply_by_10", multiply_wrapper, inject_context=True)
    graph.add_node(multiply_task, "multiply_by_10")

    # Define dependencies: lambda -> double -> multiply
    graph.add_edge("lambda_task", "double_task")
    graph.add_edge("double_task", "multiply_by_10")

    print("✅ Task graph built")

    # Step 3: Execute workflow
    print("\nStep 3: Executing workflow")
    print("⏳ Executing tasks...")

    context = ExecutionContext.create(graph, "lambda_task", max_steps=10)
    engine = WorkflowEngine()
    engine.execute(context)

    # Get results
    lambda_result = context.get_result("lambda_task")
    double_result = context.get_result("double_task")
    multiply_result = context.get_result("multiply_by_10")

    print(f"✅ lambda_task executed -> {lambda_result}")
    print(f"✅ double_task executed -> {double_result}")
    print(f"✅ multiply_by_10 executed -> {multiply_result}")

    # Step 4: Display results
    print("\nStep 4: Results")
    print(f"Lambda result: {lambda_result}")
    print(f"Double result: {double_result}")
    print(f"Multiply by 10 result: {multiply_result}")

    # Verify correctness
    assert lambda_result == 42
    assert double_result == 84
    assert multiply_result == 840

    # Summary
    print("\n=== Summary ===")
    print("✅ Lambda tasks work with cloudpickle")
    print("✅ Closures captured outer scope correctly")
    print("✅ All tasks serialized and executed")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Lambda Tasks**
#    - Can be wrapped with TaskWrapper
#    - Cloudpickle handles serialization
#    - Useful for simple inline operations
#
# 2. **Closures**
#    - Functions that capture variables from outer scope
#    - Factory pattern: create_multiplier(10) returns a function
#    - Cloudpickle serializes both function and captured variables
#
# 3. **Serialization**
#    - Standard pickle fails with lambdas and closures
#    - Cloudpickle extends pickle to handle these cases
#    - Graflow uses cloudpickle automatically
#
# 4. **Use Cases**
#    ✅ Dynamic task generation
#    ✅ Configuration-based workflows
#    ✅ Parameterized operations
#    ✅ Functional programming patterns
#
# 5. **Best Practices**
#    - Use closures for parameterized operations
#    - Factory functions for task generators
#    - Keep captured scope minimal
#    - Test serialization in distributed environments
#
# 6. **Limitations**
#    ⚠️  Captured objects must be serializable
#    ⚠️  Avoid capturing large objects
#    ⚠️  Be careful with module-level imports in closures
#    ⚠️  Some system resources cannot be serialized
#
# ============================================================================
# Advanced Patterns:
# ============================================================================
#
# **Parameterized Task Factory**:
# def create_processor(operation, multiplier):
#     def process(data):
#         if operation == "scale":
#             return [x * multiplier for x in data]
#         elif operation == "filter":
#             return [x for x in data if x > multiplier]
#     return process
#
# **Configuration-Driven Workflow**:
# config = {"factor": 10, "operation": "multiply"}
# task = TaskWrapper("configured", create_multiplier(config["factor"]))
#
# **Currying Pattern**:
# def curry_add(x):
#     def add_x(y):
#         return x + y
#     return add_x
#
# add_five = curry_add(5)
# task = TaskWrapper("add_five", lambda: add_five(10))  # Returns 15
#
# ============================================================================
