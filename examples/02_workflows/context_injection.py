"""
Context Injection Example
==========================

This example demonstrates how to access the ExecutionContext within tasks
using the inject_context parameter. This gives you access to:
- Session information (session_id, workflow_name)
- Channels for inter-task communication
- Task results storage
- Workflow execution state

Concepts Covered:
-----------------
1. Using inject_context=True to receive ExecutionContext
2. Accessing session information
3. Using channels for inter-task data sharing
4. Storing and retrieving task results
5. Workflow state inspection

Expected Output:
----------------
=== Context Injection Demo ===

Starting execution from: setup
📋 Setup Task
   Session ID: [generated-uuid]
   Task ID: setup
   Stored config in channel

🔍 Process Task
   Retrieved config: {'source': 'database', 'batch_size': 100}
   Processing with batch size: 100
   Stored metrics in channel

📊 Report Task
   Retrieved metrics: {'processed': 500, 'errors': 0}
   === Execution Report ===
   Session: [generated-uuid]
   Total processed: 500
   Total errors: 0
   Status: ✅ Success

Execution completed after 3 steps
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Demonstrate context injection in tasks."""
    print("=== Context Injection Demo ===\n")

    with workflow("data_pipeline") as ctx:

        @task(inject_context=True)
        def setup(context: TaskExecutionContext):
            """
            Setup task that stores configuration in the channel.

            With inject_context=True, the task receives the ExecutionContext
            as its first parameter.
            """
            print("📋 Setup Task")
            print(f"   Session ID: {context.session_id}")
            print(f"   Task ID: {context.task_id}")

            # Store configuration data in the channel
            # Channels allow tasks to share data without passing parameters
            config = {"source": "database", "batch_size": 100}
            channel = context.get_channel()
            channel.set("config", config)
            print("   Stored config in channel\n")

            return "setup_complete"

        @task(inject_context=True)
        def process(context: TaskExecutionContext):
            """
            Process task that retrieves config from channel and stores metrics.
            """
            print("🔍 Process Task")

            # Retrieve configuration from channel
            channel = context.get_channel()
            config = channel.get("config")
            print(f"   Retrieved config: {config}")

            batch_size = config["batch_size"]
            print(f"   Processing with batch size: {batch_size}")

            # Simulate processing and store metrics
            metrics = {"processed": 500, "errors": 0}
            channel.set("metrics", metrics)
            print("   Stored metrics in channel\n")

            return "processing_complete"

        @task(inject_context=True)
        def report(context: TaskExecutionContext):
            """
            Report task that generates a summary using channel data.
            """
            print("📊 Report Task")

            # Retrieve metrics from channel
            channel = context.get_channel()
            metrics = channel.get("metrics")
            print(f"   Retrieved metrics: {metrics}")

            # Generate report
            print("   === Execution Report ===")
            print(f"   Session: {context.session_id}")
            print(f"   Total processed: {metrics['processed']}")
            print(f"   Total errors: {metrics['errors']}")
            print("   Status: ✅ Success\n")

            return "report_complete"

        # Define workflow: setup -> process -> report
        setup >> process >> report  # type: ignore

        # Execute the workflow
        ctx.execute("setup")

    print("Context injection demo completed! 🎉")


def demo_keyword_args_resolution():
    """Demonstrate automatic keyword argument resolution from channel.

    This example shows how tasks can automatically resolve their keyword
    arguments from the channel without explicitly retrieving them.
    """
    print("\n\n" + "=" * 80)
    print("=== Automatic Keyword Argument Resolution Demo ===")
    print("=" * 80 + "\n")

    with workflow("keyword_args_pipeline") as ctx:

        @task(inject_context=True)
        def setup(context: TaskExecutionContext):
            """Setup task that stores data in channel."""
            print("📋 Setup Task")
            print("   Storing data in channel...")

            channel = context.get_channel()
            channel.set("user_name", "Alice")
            channel.set("user_age", 30)
            channel.set("user_city", "Tokyo")
            print("   ✓ Data stored\n")

        @task
        def greet(user_name: str, user_city: str = "Unknown"):
            """
            Greeting task with automatic keyword argument resolution.

            Note: No inject_context needed! The parameters user_name and user_city
            are automatically resolved from the channel.
            """
            print("👋 Greet Task")
            print(f"   Hello, {user_name} from {user_city}!")
            print("   (Arguments resolved automatically from channel)\n")

        @task(inject_context=True)
        def process_user(context: TaskExecutionContext, user_name: str, user_age: int):
            """
            Processing task combining context injection with keyword resolution.

            This task shows that you can use both inject_context=True and
            automatic keyword argument resolution together!
            """
            print("🔧 Process Task")
            print(f"   Session: {context.session_id[:8]}...")
            print(f"   Processing user: {user_name}, age {user_age}")
            print("   (Context injected + arguments resolved from channel)\n")

            # Store processing result
            channel = context.get_channel()
            channel.set("processed", True)

        @task(inject_context=True)
        def report(context: TaskExecutionContext):
            """Report task that checks processing status."""
            print("📊 Report Task")

            channel = context.get_channel()
            processed = channel.get("processed")
            user_name = channel.get("user_name")

            print(f"   User '{user_name}' processing status: {processed}")
            print("   ✅ Workflow completed\n")

        # Define workflow
        setup >> greet >> process_user >> report  # type: ignore

        # Execute
        ctx.execute("setup")

    print("Keyword argument resolution demo completed! 🎉\n")


if __name__ == "__main__":
    # Run main context injection demo
    main()

    # Run keyword argument resolution demo
    demo_keyword_args_resolution()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Enabling Context Injection**
#    @task(inject_context=True)
#    - Task receives ExecutionContext as first parameter
#    - Must declare context parameter in function signature
#    - Type hint with TaskExecutionContext for IDE support
#
# 2. **TaskExecutionContext Properties**
#    - context.session_id: Unique identifier for this execution
#    - context.task_id: ID of the current task
#    - context.get_channel(): Get channel for inter-task communication
#    - context.execution_context: Reference to parent ExecutionContext
#    - context.cycle_count: Current cycle count for this task
#    - context.max_cycles: Maximum cycles allowed for this task
#
# 3. **Channel Usage**
#    - channel = context.get_channel(): Get the channel instance
#    - channel.set(key, value): Store data
#    - channel.get(key): Retrieve data
#    - channel.keys(): List all keys
#    - Channels persist across tasks within the same workflow
#
# 4. **Automatic Keyword Argument Resolution (NEW!)**
#    @task  # No inject_context needed!
#    def my_task(param1: str, param2: int = 10):
#        # param1 and param2 automatically resolved from channel
#        pass
#
#    - Enabled by default with resolve_keyword_args=True
#    - Matches parameter names to channel keys
#    - Works with default values
#    - Can be combined with inject_context
#    - Disable with @task(resolve_keyword_args=False)
#
#    Priority order (high to low):
#    1. User-provided arguments
#    2. Injected arguments (inject_context, inject_llm_client, etc.)
#    3. Channel-resolved arguments
#
# 5. **Benefits of Context Injection**
#    ✅ Access to workflow execution state
#    ✅ Inter-task communication without parameters
#    ✅ Shared configuration and state
#    ✅ Session tracking and debugging
#    ✅ Dynamic workflow control
#
# 6. **When to Use**
#    - Tasks need to share state without direct parameter passing
#    - Need access to session/workflow metadata
#    - Want to store metrics or logs
#    - Building dynamic workflows
#    - Debugging and monitoring
#    - Automatic parameter resolution from channel (NEW!)
#
# 7. **When NOT to Use**
#    - Simple tasks with clear input/output
#    - When parameter passing is clearer
#    - Tasks that should be context-independent
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Store and retrieve different data types:
#    channel = context.get_channel()
#    channel.set("data", [1, 2, 3])
#    channel.set("config", {"key": "value"})
#    channel.set("result", 42)
#
# 2. Combine context injection with automatic argument resolution:
#    @task(inject_context=True)
#    def my_task(context: TaskExecutionContext, name: str, count: int = 5):
#        # context: injected
#        # name, count: automatically resolved from channel
#        channel = context.get_channel()
#        print(f"Processing {name} with count {count}")
#
# 3. Use keyword argument resolution without inject_context:
#    @task
#    def process_data(input_file: str, output_dir: str = "/tmp"):
#        # Both parameters automatically resolved from channel
#        # No need to inject context!
#        pass
#
# 4. Check task state:
#    print(f"Task ID: {context.task_id}")
#    print(f"Cycle count: {context.cycle_count}/{context.max_cycles}")
#    print(f"Elapsed time: {context.elapsed_time()}")
#
# 5. Implement error tracking:
#    channel = context.get_channel()
#    errors = channel.get("errors", [])
#    errors.append("Some error")
#    channel.set("errors", errors)
#
# 6. Build a metrics collector:
#    @task(inject_context=True)
#    def track_metrics(context: TaskExecutionContext):
#        channel = context.get_channel()
#        metrics = channel.get("metrics", {})
#        metrics["timestamp"] = time.time()
#        channel.set("metrics", metrics)
#
# 7. Disable automatic resolution when needed:
#    @task(resolve_keyword_args=False)
#    def legacy_task(value: str = "default"):
#        # Won't resolve from channel, uses default
#        pass
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Configuration Management**:
# - Load config in setup task
# - Access config in all subsequent tasks
# - No need to pass config as parameters everywhere
#
# **Metrics Collection**:
# - Each task stores its metrics in channel
# - Final task aggregates and reports all metrics
# - Centralized monitoring
#
# **Error Handling**:
# - Tasks store errors in channel
# - Error handler task checks for errors
# - Conditional execution based on error state
#
# **Shared Resources**:
# - Setup task initializes resources (DB connection, API client)
# - All tasks access shared resources via channel
# - Cleanup task closes resources
#
# **Dynamic Workflows**:
# - Tasks inspect context.graph
# - Conditionally add/skip tasks
# - Workflow behavior adapts to runtime state
#
# ============================================================================
