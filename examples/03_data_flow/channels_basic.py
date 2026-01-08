"""
Basic Channels Example
=======================

This example demonstrates the fundamental operations of channels in Graflow.
Channels provide a way for tasks to communicate and share data without direct
parameter passing, which is especially useful for:
- Shared configuration
- Accumulating state
- Broadcasting data
- Decoupling task dependencies

Concepts Covered:
-----------------
1. Getting channel instances from TaskExecutionContext
2. Setting values in channels
3. Getting values from channels (with defaults)
4. Checking key existence
5. Listing all channel keys
6. Channel data persistence across tasks

Expected Output:
----------------
=== Basic Channels Demo ===

Starting execution from: producer_1
📤 Producer 1: Writing data to channel
   Set 'config' = {'batch_size': 100, 'timeout': 30}
   Set 'counter' = 1

📤 Producer 2: Writing more data to channel
   Current counter: 1
   Set 'counter' = 2
   Set 'status' = initialized

📥 Consumer: Reading data from channel
   Available keys: ['config', 'counter', 'status']
   Config: {'batch_size': 100, 'timeout': 30}
   Counter: 2
   Status: initialized
   Missing key (with default): default_value

Execution completed after 3 steps

All tasks successfully communicated via channels! 🎉
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Demonstrate basic channel operations."""
    print("=== Basic Channels Demo ===\n")

    with workflow("channel_demo") as ctx:

        @task(inject_context=True)
        def producer_1(context: TaskExecutionContext):
            """First task that writes data to the channel."""
            print("📤 Producer 1: Writing data to channel")

            # Get the channel instance
            channel = context.get_channel()

            # Set configuration data
            config = {"batch_size": 100, "timeout": 30}
            channel.set("config", config)
            print(f"   Set 'config' = {config}")

            # Set a counter
            channel.set("counter", 1)
            print("   Set 'counter' = 1\n")

        @task(inject_context=True)
        def producer_2(context: TaskExecutionContext):
            """Second task that reads and updates channel data."""
            print("📤 Producer 2: Writing more data to channel")

            channel = context.get_channel()

            # Read existing counter
            counter = channel.get("counter")
            print(f"   Current counter: {counter}")

            # Increment and update
            channel.set("counter", counter + 1)
            print(f"   Set 'counter' = {counter + 1}")

            # Add status information
            channel.set("status", "initialized")
            print("   Set 'status' = initialized\n")

        @task(inject_context=True)
        def consumer(context: TaskExecutionContext):
            """Consumer task that reads all data from the channel."""
            print("📥 Consumer: Reading data from channel")

            channel = context.get_channel()

            # List all available keys
            keys = channel.keys()
            print(f"   Available keys: {sorted(keys)}")

            # Read configuration
            config = channel.get("config")
            print(f"   Config: {config}")

            # Read counter
            counter = channel.get("counter")
            print(f"   Counter: {counter}")

            # Read status
            status = channel.get("status")
            print(f"   Status: {status}")

            # Try to read a non-existent key with a default value
            missing = channel.get("non_existent_key", default="default_value")
            print(f"   Missing key (with default): {missing}\n")

        # Define the workflow: producer_1 -> producer_2 -> consumer
        producer_1 >> producer_2 >> consumer

        # Execute the workflow
        ctx.execute("producer_1")

    print("All tasks successfully communicated via channels! 🎉")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Getting the Channel**
#    channel = context.get_channel()
#    - Every task with inject_context=True can access the channel
#    - The same channel instance is shared across all tasks in a workflow
#
# 2. **Setting Values**
#    channel.set(key, value)
#    - Store any Python object (dict, list, int, str, custom objects)
#    - Keys are strings
#    - Values persist for the entire workflow execution
#
# 3. **Getting Values**
#    value = channel.get(key, default=None)
#    - Retrieve previously stored values
#    - Always provide a default for optional keys
#    - Returns None (or default) if key doesn't exist
#
# 4. **Checking Key Existence**
#    if "key" in channel.keys():
#        value = channel.get("key")
#    - Use channel.keys() to list all available keys
#    - Check before getting to avoid None values
#
# 5. **Channel Lifecycle**
#    - Channel data persists for the entire workflow execution
#    - All tasks within the same workflow share the same channel
#    - Channel is cleared when workflow completes
#
# 6. **When to Use Channels**
#    ✅ Shared configuration across multiple tasks
#    ✅ Accumulating state (counters, metrics, logs)
#    ✅ Broadcasting data to multiple consumers
#    ✅ Decoupling producer and consumer tasks
#
# 7. **When NOT to Use Channels**
#    ❌ Direct task-to-task parameter passing (use function params)
#    ❌ Storing large datasets (use references/paths instead)
#    ❌ Required task inputs (use function parameters)
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Store different data types:
#    channel.set("list_data", [1, 2, 3])
#    channel.set("dict_data", {"a": 1, "b": 2})
#    channel.set("string_data", "hello")
#    channel.set("number_data", 42)
#
# 2. Implement a counter pattern:
#    @task(inject_context=True)
#    def increment(ctx: TaskExecutionContext):
#        channel = ctx.get_channel()
#        count = channel.get("count", 0)
#        channel.set("count", count + 1)
#
# 3. Accumulate logs:
#    logs = channel.get("logs", [])
#    logs.append(f"Task {ctx.task_id} executed")
#    channel.set("logs", logs)
#
# 4. Share configuration:
#    # Setup task
#    channel.set("config", load_config())
#
#    # All other tasks
#    config = channel.get("config")
#    use_config(config)
#
# 5. Implement feature flags:
#    channel.set("debug_mode", True)
#    if channel.get("debug_mode", False):
#        print("Debug information...")
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Configuration Management**:
# Setup task loads config once, all tasks read from channel
#
# **Metrics Collection**:
# Each task increments counters, final task reports all metrics
#
# **Error Tracking**:
# Tasks append errors to a list, error handler processes all errors
#
# **Progress Tracking**:
# Tasks update progress percentage, monitoring task displays progress
#
# **Resource Pooling**:
# Tasks share connection strings, each creates its own connection
#
# **State Machine**:
# Tasks read/write workflow state, enabling conditional execution
#
# ============================================================================
