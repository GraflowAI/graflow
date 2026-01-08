"""
Custom Handler Example
=======================

This example demonstrates how to create your own custom task execution handler.
Custom handlers allow you to control exactly how and where tasks execute:
- Remote execution (SSH, cloud, etc.)
- Specialized hardware (GPU, TPU)
- External systems (APIs, databases)
- Custom logging and monitoring
- Special execution logic

Concepts Covered:
-----------------
1. TaskHandler interface
2. Handler implementation
3. Handler registration with WorkflowEngine
4. Custom execution logic
5. Error handling in handlers
6. Result storage

Expected Output:
----------------
=== Custom Handler Demo ===

Starting execution from: normal_task
✅ Normal Task (Direct Handler)
   Executing with DirectTaskHandler

⚡ Custom Task (Logging Handler)
   [LoggingHandler] Starting execution of task: custom_task
   [LoggingHandler] Task function: custom_task
   [LoggingHandler] Execution started at: 2025-01-08 12:34:56
   Executing custom task logic
   [LoggingHandler] Execution completed at: 2025-01-08 12:34:56
   [LoggingHandler] Duration: 0.001s
   [LoggingHandler] Result: custom_result
   [LoggingHandler] Stored result in context

⚡ Timed Task (Timing Handler)
   [TimingHandler] Executing task: timed_task
   Executing timed task
   [TimingHandler] Task completed in 0.100s

✅ Results Task
   Normal task result: normal_result
   Custom task result: custom_result
   Timed task result: timed_result

Execution completed after 4 steps

=== Summary ===
Created two custom handlers:
1. LoggingHandler - Detailed execution logging
2. TimingHandler - Performance timing
Both handlers successfully executed tasks!
"""

import time
from datetime import datetime

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handler import TaskHandler
from graflow.core.task import Executable
from graflow.core.workflow import workflow

# ============================================================================
# Custom Handler Implementations
# ============================================================================


class LoggingHandler(TaskHandler):
    """Custom handler that logs detailed execution information.

    This handler wraps task execution with detailed logging:
    - Task start/end timestamps
    - Execution duration
    - Result values
    - Error information
    """

    def execute_task(self, task: Executable, context: ExecutionContext):
        """Execute task with detailed logging.

        Args:
            task: Executable task to run
            context: Execution context for result storage
        """
        task_id = task.task_id
        start_time = datetime.now()

        print(f"   [LoggingHandler] Starting execution of task: {task_id}")
        print(f"   [LoggingHandler] Task function: {task_id}")
        print(f"   [LoggingHandler] Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Execute the task
            result = task.run()

            # Log completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"   [LoggingHandler] Execution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   [LoggingHandler] Duration: {duration:.3f}s")
            print(f"   [LoggingHandler] Result: {result}")

            # Store result in context
            context.set_result(task_id, result)
            print("   [LoggingHandler] Stored result in context\n")
            return result

        except Exception as e:
            # Log error
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"   [LoggingHandler] Execution failed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   [LoggingHandler] Duration before failure: {duration:.3f}s")
            print(f"   [LoggingHandler] Error: {str(e)}\n")

            # Store exception in context
            context.set_result(task_id, e)
            raise


class TimingHandler(TaskHandler):
    """Custom handler that measures and reports execution time.

    This handler focuses on performance timing:
    - Measures precise execution duration
    - Reports timing information
    - Useful for performance profiling
    """

    def execute_task(self, task: Executable, context: ExecutionContext):
        """Execute task with timing measurement.

        Args:
            task: Executable task to run
            context: Execution context for result storage
        """
        task_id = task.task_id
        print(f"   [TimingHandler] Executing task: {task_id}")

        try:
            # Measure execution time
            start = time.perf_counter()
            result = task.run()
            elapsed = time.perf_counter() - start

            # Report timing
            print(f"   [TimingHandler] Task completed in {elapsed:.3f}s\n")

            # Store result
            context.set_result(task_id, result)
            return result

        except Exception as e:
            # Store exception
            context.set_result(task_id, e)
            raise


# ============================================================================
# Demo Workflow
# ============================================================================


def main():
    """Demonstrate custom handler usage."""
    print("=== Custom Handler Demo ===\n")

    with workflow("custom_handler_demo") as ctx:

        @task(handler="direct")
        def normal_task():
            """Normal task using DirectTaskHandler."""
            print("✅ Normal Task (Direct Handler)")
            print("   Executing with DirectTaskHandler\n")
            return "normal_result"

        @task(handler="logging")
        def custom_task():
            """Task using custom LoggingHandler."""
            print("   Executing custom task logic")
            return "custom_result"

        @task(handler="timing")
        def timed_task():
            """Task using custom TimingHandler."""
            print("   Executing timed task")
            # Simulate some work
            time.sleep(0.1)
            return "timed_result"

        @task(handler="direct", inject_context=True)
        def results_task(context):
            """Collect and display results from all tasks."""

            print("✅ Results Task")

            # Get results
            normal_result = context.get_result("normal_task")
            custom_result = context.get_result("custom_task")
            timed_result = context.get_result("timed_task")

            print(f"   Normal task result: {normal_result}")
            print(f"   Custom task result: {custom_result}")
            print(f"   Timed task result: {timed_result}\n")

            return {"normal": normal_result, "custom": custom_result, "timed": timed_result}

        # Define workflow
        normal_task >> custom_task >> timed_task >> results_task

        # Create engine and register custom handlers
        engine = WorkflowEngine()
        engine.register_handler("logging", LoggingHandler())
        engine.register_handler("timing", TimingHandler())

        # Create execution context
        exec_context = ExecutionContext.create(ctx.graph, "normal_task", max_steps=10)

        # Execute with custom engine
        engine.execute(exec_context)

    # Summary
    print("=== Summary ===")
    print("Created two custom handlers:")
    print("1. LoggingHandler - Detailed execution logging")
    print("2. TimingHandler - Performance timing")
    print("Both handlers successfully executed tasks!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **TaskHandler Interface**
#    from graflow.core.handler import TaskHandler
#
#    class MyHandler(TaskHandler):
#        def execute_task(self, task, context):
#            # Implement execution logic
#            result = task.run()
#            context.set_result(task.task_id, result)
#
#    - Inherit from TaskHandler base class
#    - Implement execute_task() method
#    - MUST call context.set_result()
#
# 2. **Handler Registration**
#    engine = WorkflowEngine()
#    engine.register_handler("my_handler", MyHandler())
#
#    - Register before executing workflow
#    - Use unique handler type name
#    - Can register multiple handlers
#
# 3. **Handler Selection**
#    @task(handler="my_handler")
#    def my_task():
#        pass
#
#    - Tasks specify handler by type name
#    - Must match registered handler name
#    - Raises error if handler not found
#
# 4. **Result Storage**
#    context.set_result(task_id, result)
#
#    - MUST store result after execution
#    - Store exceptions for error handling
#    - Other tasks can retrieve via get_result()
#
# 5. **Error Handling**
#    try:
#        result = task.run()
#        context.set_result(task_id, result)
#    except Exception as e:
#        context.set_result(task_id, e)
#        raise  # Re-raise to stop workflow
#
#    - Catch exceptions during execution
#    - Store exception in context
#    - Re-raise to propagate error
#
# 6. **Handler Use Cases**
#    ✅ Logging and monitoring
#    ✅ Performance profiling
#    ✅ Remote execution (SSH, cloud)
#    ✅ Retry logic
#    ✅ Resource management
#    ✅ External system integration
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Create a retry handler:
#    class RetryHandler(TaskHandler):
#        def __init__(self, max_retries=3):
#            self.max_retries = max_retries
#
#        def execute_task(self, task, context):
#            for attempt in range(self.max_retries):
#                try:
#                    result = task.run()
#                    context.set_result(task.task_id, result)
#                    return
#                except Exception as e:
#                    if attempt == self.max_retries - 1:
#                        context.set_result(task.task_id, e)
#                        raise
#                    print(f"Retry {attempt + 1}/{self.max_retries}")
#
# 2. Create a caching handler:
#    class CachingHandler(TaskHandler):
#        def __init__(self):
#            self.cache = {}
#
#        def execute_task(self, task, context):
#            # Check cache
#            if task.task_id in self.cache:
#                print("Cache hit!")
#                context.set_result(task.task_id, self.cache[task.task_id])
#                return
#
#            # Execute and cache
#            result = task.run()
#            self.cache[task.task_id] = result
#            context.set_result(task.task_id, result)
#
# 3. Create a rate-limiting handler:
#    import time
#
#    class RateLimitHandler(TaskHandler):
#        def __init__(self, min_interval=1.0):
#            self.min_interval = min_interval
#            self.last_execution = 0
#
#        def execute_task(self, task, context):
#            # Wait if needed
#            elapsed = time.time() - self.last_execution
#            if elapsed < self.min_interval:
#                time.sleep(self.min_interval - elapsed)
#
#            # Execute
#            result = task.run()
#            self.last_execution = time.time()
#            context.set_result(task.task_id, result)
#
# 4. Create a metrics handler:
#    class MetricsHandler(TaskHandler):
#        def __init__(self):
#            self.metrics = []
#
#        def execute_task(self, task, context):
#            start = time.time()
#            try:
#                result = task.run()
#                status = "success"
#                context.set_result(task.task_id, result)
#            except Exception as e:
#                status = "failed"
#                context.set_result(task.task_id, e)
#                raise
#            finally:
#                self.metrics.append({
#                    "task": task.task_id,
#                    "duration": time.time() - start,
#                    "status": status
#                })
#
# 5. Create an async handler:
#    import asyncio
#
#    class AsyncHandler(TaskHandler):
#        def execute_task(self, task, context):
#            # Run task in async context
#            loop = asyncio.new_event_loop()
#            asyncio.set_event_loop(loop)
#            try:
#                result = loop.run_until_complete(task.run())
#                context.set_result(task.task_id, result)
#            finally:
#                loop.close()
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **SSH Execution Handler**:
# Execute tasks on remote machines via SSH
#
# **Cloud Function Handler**:
# Execute tasks as AWS Lambda, Google Cloud Functions, etc.
#
# **GPU Queue Handler**:
# Queue tasks for execution on available GPUs
#
# **Database Handler**:
# Store task code in DB, execute by workers, store results in DB
#
# **Webhook Handler**:
# Send task to external service via HTTP POST
#
# **Monitoring Handler**:
# Wrap tasks with detailed metrics collection and reporting
#
# ============================================================================
# Advanced Patterns:
# ============================================================================
#
# **Composite Handler (Decorator Pattern)**:
# class MonitoringWrapper(TaskHandler):
#     def __init__(self, base_handler):
#         self.base_handler = base_handler
#
#     def execute_task(self, task, context):
#         # Add monitoring around base handler
#         start = time.time()
#         self.base_handler.execute_task(task, context)
#         print(f"Monitored: {time.time() - start}s")
#
# **Conditional Handler**:
# class ConditionalHandler(TaskHandler):
#     def execute_task(self, task, context):
#         # Check condition
#         if should_execute(task):
#             result = task.run()
#         else:
#             result = None  # Skip execution
#         context.set_result(task.task_id, result)
#
# **Parallel Handler**:
# class ParallelHandler(TaskHandler):
#     def execute_task(self, task, context):
#         import concurrent.futures
#
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(task.run)
#             result = future.result()
#             context.set_result(task.task_id, result)
#
# ============================================================================
