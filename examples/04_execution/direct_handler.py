"""
Direct Handler Example
=======================

This example demonstrates the DirectTaskHandler - the default execution handler
in Graflow. The DirectTaskHandler executes tasks in-process within the same
Python interpreter, providing the fastest execution with no overhead.

Concepts Covered:
-----------------
1. Default handler behavior
2. Explicit handler specification
3. In-process execution characteristics
4. Performance benefits
5. When to use DirectTaskHandler

Expected Output:
----------------
=== Direct Handler Demo ===

Starting execution from: task_default
✅ Default Task (no handler specified)
   Execution environment: Same process as main
   Handler type: direct (default)

✅ Explicit Direct Task
   Execution environment: Same process as main
   Handler type: direct (explicit)

✅ Fast Computation Task
   Computing Fibonacci(30)...
   Result: 832040
   Execution time: ~0.XXX seconds

Execution completed after 3 steps

=== Summary ===
All tasks executed in-process with DirectTaskHandler
- No containerization overhead
- Shared memory space
- Fast execution
- Perfect for most use cases! ✅
"""

import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def fibonacci(n: int) -> int:
    """Calculate Fibonacci number (recursive for demonstration)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    """Demonstrate DirectTaskHandler usage."""
    print("=== Direct Handler Demo ===\n")

    with workflow("direct_handler_demo") as ctx:

        @task
        def task_default():
            """
            Task without explicit handler specification.

            When no handler is specified, DirectTaskHandler is used by default.
            This is the recommended approach for most tasks.
            """
            print("✅ Default Task (no handler specified)")
            print("   Execution environment: Same process as main")
            print("   Handler type: direct (default)\n")
            return "default_result"

        @task(handler="direct")
        def task_explicit():
            """
            Task with explicit handler="direct" specification.

            This is equivalent to omitting the handler parameter.
            Use this when you want to be explicit about execution mode.
            """
            print("✅ Explicit Direct Task")
            print("   Execution environment: Same process as main")
            print("   Handler type: direct (explicit)\n")
            return "explicit_result"

        @task(handler="direct")
        def task_performance():
            """
            Demonstrate the performance benefits of DirectTaskHandler.

            DirectTaskHandler has:
            - No containerization overhead
            - No serialization/deserialization
            - Native Python performance
            """
            print("✅ Fast Computation Task")
            print("   Computing Fibonacci(30)...")

            start = time.time()
            result = fibonacci(30)
            elapsed = time.time() - start

            print(f"   Result: {result}")
            print(f"   Execution time: ~{elapsed:.3f} seconds\n")

            return result

        # Define workflow
        task_default >> task_explicit >> task_performance

        # Execute
        ctx.execute("task_default")

    # Summary
    print("=== Summary ===")
    print("All tasks executed in-process with DirectTaskHandler")
    print("- No containerization overhead")
    print("- Shared memory space")
    print("- Fast execution")
    print("- Perfect for most use cases! ✅")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Default Handler**
#    @task
#    def my_task():
#        pass
#
#    - DirectTaskHandler is the default when no handler is specified
#    - No need to explicitly set handler="direct" in most cases
#    - Simplest and most common execution mode
#
# 2. **Explicit Specification**
#    @task(handler="direct")
#    def my_task():
#        pass
#
#    - Equivalent to omitting the handler parameter
#    - Use when you want to be explicit about execution mode
#    - Useful for documentation and clarity
#
# 3. **In-Process Execution**
#    - Tasks run in the same Python process
#    - Shares memory space with main program
#    - Can access module-level variables
#    - No isolation between tasks
#
# 4. **Performance Characteristics**
#    - Fastest execution (no overhead)
#    - No serialization/deserialization
#    - No container startup time
#    - Native Python performance
#
# 5. **When to Use DirectTaskHandler**
#    ✅ Trusted code
#    ✅ Local development
#    ✅ Fast iteration
#    ✅ Most production tasks
#    ✅ Tasks that don't need isolation
#    ✅ Performance-critical operations
#
# 6. **When NOT to Use DirectTaskHandler**
#    ❌ Untrusted code (use Docker)
#    ❌ Need process isolation (use Docker)
#    ❌ Different Python versions (use Docker)
#    ❌ Specific system dependencies (use Docker)
#    ❌ Remote execution (use custom handler)
#
# 7. **Shared State**
#    # Module-level variables are shared
#    counter = 0
#
#    @task
#    def increment():
#        global counter
#        counter += 1  # Modifies shared state
#
#    @task
#    def read():
#        return counter  # Reads shared state
#
#    - Be careful with shared mutable state
#    - Use channels for explicit communication
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Compare with Docker handler (next example):
#    @task(handler="direct")
#    def fast():
#        return "instant"
#
#    @task(handler="docker")  # Much slower
#    def slow():
#        return "delayed"
#
# 2. Test shared state:
#    global_list = []
#
#    @task
#    def append_task():
#        global_list.append(1)
#
#    @task
#    def read_task():
#        return len(global_list)
#
# 3. Measure execution overhead:
#    import time
#
#    @task
#    def timed_task():
#        start = time.time()
#        # Do work
#        return time.time() - start
#
# 4. Access imported modules:
#    import numpy as np
#
#    @task
#    def use_numpy():
#        return np.array([1, 2, 3]).sum()
#
# 5. Use closures:
#    def create_task(multiplier):
#        @task
#        def multiply(x):
#            return x * multiplier
#        return multiply
#
#    task_double = create_task(2)
#    result = task_double(5)  # Returns 10
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Data Processing Pipeline**:
# All tasks run in-process for maximum performance
#
# **Web API Backend**:
# Task-based request handlers with fast execution
#
# **Local Machine Learning**:
# Training and inference tasks without containerization overhead
#
# **Database Operations**:
# Query execution tasks with shared connection pool
#
# **File Processing**:
# Read, transform, write tasks with direct filesystem access
#
# **Real-time Systems**:
# Low-latency task execution for time-critical operations
#
# ============================================================================
# Advanced Topics:
# ============================================================================
#
# **Memory Sharing**:
# @task
# def producer():
#     return large_dataset  # Passed by reference (no copy)
#
# @task
# def consumer(data):
#     # Receives reference to same object
#     return process(data)
#
# **Exception Handling**:
# @task
# def may_fail():
#     try:
#         risky_operation()
#     except Exception as e:
#         # Exception is caught and stored in context
#         raise
#
# **Resource Management**:
# @task
# def use_resources():
#     with open("file.txt") as f:
#         return f.read()
#     # File automatically closed
#
# **Threading Compatibility**:
# @task
# def threaded():
#     import threading
#     # Can use threading within tasks
#     thread = threading.Thread(target=worker)
#     thread.start()
#     thread.join()
#
# ============================================================================
# Comparison: Direct vs Other Handlers
# ============================================================================
#
# | Feature | Direct | Docker | Custom |
# |---------|--------|--------|--------|
# | Speed | Fastest | Slow | Varies |
# | Isolation | None | Full | Configurable |
# | Overhead | ~0ms | ~500-2000ms | Varies |
# | Setup | None | Docker required | Custom |
# | Use Case | Most tasks | Isolation | Special needs |
#
# **Recommendation**: Start with DirectTaskHandler for all tasks.
# Only use other handlers when you have a specific need for:
# - Process isolation (Docker)
# - Remote execution (Custom)
# - Different environments (Docker)
#
# ============================================================================
