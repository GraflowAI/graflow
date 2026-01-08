"""
Runtime Dynamic Task Generation
================================

This example demonstrates runtime dynamic task generation using context.next_task()
and context.next_iteration(). Unlike compile-time task generation (using loops and
factories), these methods create tasks during workflow execution based on runtime
conditions and data.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Runtime task creation with context.next_task()
2. Iterative processing with context.next_iteration()
3. Conditional task branching at runtime
4. TaskWrapper for dynamic task creation
5. Data-driven workflow adaptation

Expected Output:
----------------
=== Runtime Dynamic Task Generation ===

Scenario 1: Conditional Task Creation

Processing data: value=150
✅ Value > 100, creating high-value task
High-value processing: 150 → 300

Processing data: value=50
✅ Value <= 100, creating standard task
Standard processing: 50 → 60

Scenario 2: Iterative Processing

Iteration 0: accuracy=0.50
Iteration 1: accuracy=0.65
Iteration 2: accuracy=0.80
Iteration 3: accuracy=0.95
✅ Converged! Saving final model

Scenario 3: Multi-Level Processing

Batch 1: Processing 3 items
  Item 0: value=10 → result=20
  Item 1: value=20 → result=40
  Item 2: value=30 → result=60
Aggregating 3 results: total=120

=== Summary ===
✅ Runtime task creation demonstrated
✅ Iterative processing with convergence
✅ Conditional branching at runtime
✅ Data-driven workflow adaptation
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import TaskWrapper
from graflow.core.workflow import workflow


def scenario_1_conditional_tasks():
    """Scenario 1: Create different tasks based on runtime conditions."""
    print("=== Scenario 1: Conditional Task Creation ===\n")

    def process_high_value(value):
        """Process high-value data."""
        result = value * 2
        print(f"High-value processing: {value} → {result}\n")
        return {"type": "high", "result": result}

    def process_standard_value(value):
        """Process standard-value data."""
        result = value + 10
        print(f"Standard processing: {value} → {result}\n")
        return {"type": "standard", "result": result}

    with workflow("conditional_tasks") as ctx:

        @task(id="classifier", inject_context=True)
        def classify_and_process(context: TaskExecutionContext):
            """Classify data and create appropriate processing task."""
            # Simulate different data scenarios
            test_values = [150, 50]

            for value in test_values:
                print(f"Processing data: value={value}")

                if value > 100:
                    # Create high-value processing task
                    print("✅ Value > 100, creating high-value task")
                    high_task = TaskWrapper(f"high_processor_{value}", lambda v=value: process_high_value(v))
                    context.next_task(high_task)
                else:
                    # Create standard processing task
                    print("✅ Value <= 100, creating standard task")
                    std_task = TaskWrapper(f"std_processor_{value}", lambda v=value: process_standard_value(v))
                    context.next_task(std_task)

            return {"processed": len(test_values)}

        ctx.execute("classifier", max_steps=10)


def scenario_2_iterative_processing():
    """Scenario 2: Iterative processing with convergence checking."""
    print("\n=== Scenario 2: Iterative Processing ===\n")

    def save_final_model(params):
        """Save the optimized model."""
        print("✅ Converged! Saving final model\n")
        return {"saved": True, "final_accuracy": params["accuracy"], "iterations": params["iteration"]}

    with workflow("iterative_optimization") as ctx:

        @task(id="optimizer", inject_context=True)
        def optimize(context: TaskExecutionContext, data=None):
            """Optimize parameters iteratively until convergence."""
            # Get parameters from data or initialize
            if data is None:
                params = {"iteration": 0, "accuracy": 0.5, "learning_rate": 0.1}
            else:
                params = data

            iteration = params["iteration"]
            accuracy = params["accuracy"]

            print(f"Iteration {iteration}: accuracy={accuracy:.2f}")

            # Simulate optimization step
            new_accuracy = min(accuracy + 0.15, 0.95)
            updated_params = {
                "iteration": iteration + 1,
                "accuracy": new_accuracy,
                "learning_rate": params["learning_rate"] * 0.95,
            }

            # Check convergence
            if new_accuracy >= 0.9:
                # Converged - create final task
                final_task = TaskWrapper("save_model", lambda: save_final_model(updated_params))
                context.next_task(final_task)
            else:
                # Continue optimization
                context.next_iteration(updated_params)

            return updated_params

        ctx.execute("optimizer", max_steps=10)


def scenario_3_batch_processing():
    """Scenario 3: Dynamic parallel task generation for batch processing."""
    print("\n=== Scenario 3: Multi-Level Processing ===\n")

    def process_item(item_id, value):
        """Process a single batch item."""
        result = value * 2
        print(f"  Item {item_id}: value={value} → result={result}")
        return {"item_id": item_id, "result": result}

    def aggregate_results(count, total):
        """Aggregate results from all items."""
        print(f"Aggregating {count} results: total={total}\n")
        return {"count": count, "total": total}

    with workflow("batch_processing") as ctx:

        @task(id="batch_coordinator", inject_context=True)
        def coordinate_batch(context: TaskExecutionContext):
            """Coordinate processing of batch items."""
            batch_items = [10, 20, 30]
            batch_id = 1

            print(f"Batch {batch_id}: Processing {len(batch_items)} items")

            # Create tasks for each item
            results = []
            for i, value in enumerate(batch_items):
                # Process synchronously for simplicity
                result = process_item(i, value)
                results.append(result)

            # Calculate totals
            total = sum(r["result"] for r in results)

            # Create aggregation task
            agg_task = TaskWrapper(f"aggregator_{batch_id}", lambda: aggregate_results(len(results), total))
            context.next_task(agg_task)

            return {"batch_id": batch_id, "items_processed": len(batch_items)}

        ctx.execute("batch_coordinator", max_steps=10)


def scenario_4_error_recovery():
    """Scenario 4: Error handling with retry using next_iteration."""
    print("\n=== Scenario 4: Error Recovery with Retry ===\n")

    def risky_operation(attempt):
        """Simulate an operation that might fail."""
        import random

        success = random.random() > 0.3 or attempt >= 2
        if success:
            print(f"  ✅ Operation succeeded on attempt {attempt + 1}")
            return {"success": True, "attempt": attempt}
        else:
            print(f"  ❌ Operation failed on attempt {attempt + 1}")
            raise Exception("Simulated failure")

    with workflow("error_recovery") as ctx:

        @task(id="resilient_task", inject_context=True)
        def process_with_retry(context: TaskExecutionContext):
            """Process with automatic retry on failure."""
            channel = context.get_channel()
            attempt = channel.get("attempt", default=0)
            max_retries = 3

            print(f"Attempt {attempt + 1}/{max_retries}")

            try:
                # Try the risky operation
                result = risky_operation(attempt)
                print("✅ Task completed successfully\n")
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    # Retry
                    print(f"  🔄 Retrying... ({attempt + 2}/{max_retries})\n")
                    channel.set("attempt", attempt + 1)
                    context.next_iteration()
                else:
                    # Max retries reached
                    print("  ⚠️  Max retries reached, giving up\n")
                    return {"success": False, "error": str(e)}

        ctx.execute("resilient_task", max_steps=10)


def scenario_5_state_machine():
    """Scenario 5: State machine with runtime transitions."""
    print("\n=== Scenario 5: State Machine ===\n")

    def process_state(state, data):
        """Process current state."""
        print(f"  Processing state: {state} with data: {data}")
        return {"state": state, "processed": True}

    with workflow("state_machine") as ctx:

        @task(id="state_controller", inject_context=True)
        def control_states(context: TaskExecutionContext):
            """Control state transitions."""
            channel = context.get_channel()
            current_state = channel.get("state", default="START")
            data = channel.get("data", default=0)

            print(f"Current state: {current_state} (data={data})")

            # State transitions
            if current_state == "START":
                # Transition to PROCESSING
                print("→ Transitioning to PROCESSING")
                channel.set("state", "PROCESSING")
                channel.set("data", data + 1)
                context.next_iteration()

            elif current_state == "PROCESSING":
                if data < 3:
                    # Stay in PROCESSING
                    print("→ Continuing PROCESSING")
                    channel.set("data", data + 1)
                    context.next_iteration()
                else:
                    # Transition to FINALIZING
                    print("→ Transitioning to FINALIZING")
                    channel.set("state", "FINALIZING")
                    context.next_iteration()

            elif current_state == "FINALIZING":
                # Transition to END
                print("→ Transitioning to END")
                channel.set("state", "END")
                # Create final task
                final_task = TaskWrapper("end_state", lambda: process_state("END", data))
                context.next_task(final_task)

            return {"state": current_state, "data": data}

        ctx.execute("state_controller", max_steps=15)
        print()


def main():
    """Run all runtime dynamic task scenarios."""
    print("=== Runtime Dynamic Task Generation ===\n")

    # Scenario 1: Conditional task creation
    scenario_1_conditional_tasks()

    # Scenario 2: Iterative processing
    scenario_2_iterative_processing()

    # Scenario 3: Batch processing
    scenario_3_batch_processing()

    # Scenario 4: Error recovery
    scenario_4_error_recovery()

    # Scenario 5: State machine
    scenario_5_state_machine()

    print("=== Summary ===")
    print("✅ Runtime task creation demonstrated")
    print("✅ Iterative processing with convergence")
    print("✅ Conditional branching at runtime")
    print("✅ Data-driven workflow adaptation")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Runtime Dynamic Task Creation**
#    - Use context.next_task(TaskWrapper(...)) to create tasks at runtime
#    - Tasks are created based on execution results and conditions
#    - Different from compile-time task generation (loops, factories)
#
# 2. **TaskWrapper for Dynamic Tasks**
#    task = TaskWrapper("task_id", function)
#    context.next_task(task)
#    - TaskWrapper wraps a function as an executable task
#    - Requires unique task_id
#    - Function can be lambda or regular function
#
# 3. **Iterative Processing**
#    context.next_iteration(data)
#    - Creates another iteration of the current task
#    - Pass updated data/state to next iteration
#    - Use for convergence, retry, loops
#
# 4. **When to Use Runtime Dynamic Tasks**
#    ✅ Conditional workflow branching based on data
#    ✅ Iterative optimization algorithms
#    ✅ Retry and error recovery patterns
#    ✅ State machines
#    ✅ Adaptive workflows
#
# 5. **Compile-Time vs Runtime Task Generation**
#    Compile-Time (see dynamic_tasks.py):
#    - Tasks defined before execution with @task
#    - Generated in loops/factories
#    - Structure known upfront
#
#    Runtime (this example):
#    - Tasks created during execution
#    - Based on runtime conditions/data
#    - Structure emerges during execution
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **ML Hyperparameter Tuning**:
# @task(inject_context=True)
# def tune_hyperparameters(context):
#     params = optimize_step()
#     if not converged(params):
#         context.next_iteration(params)
#     else:
#         save_task = TaskWrapper("save_model", lambda: save(params))
#         context.next_task(save_task)
#
# **Data Classification Pipeline**:
# @task(inject_context=True)
# def classify_and_route(context):
#     data_type = detect_type(data)
#     if data_type == "image":
#         task = TaskWrapper("image_proc", process_image)
#     elif data_type == "text":
#         task = TaskWrapper("text_proc", process_text)
#     context.next_task(task)
#
# **Resilient API Calls**:
# @task(inject_context=True)
# def call_api_with_retry(context):
#     try:
#         result = api_call()
#         return result
#     except Exception:
#         if attempts < max_retries:
#             context.next_iteration({"attempts": attempts + 1})
#         else:
#             error_task = TaskWrapper("log_error", handle_error)
#             context.next_task(error_task)
#
# **Progressive Enhancement**:
# @task(inject_context=True)
# def enhance_quality(context):
#     quality = current_quality()
#     if quality < target_quality:
#         enhanced = apply_enhancement()
#         context.next_iteration({"quality": enhanced})
#     else:
#         done_task = TaskWrapper("finalize", save_result)
#         context.next_task(done_task)
#
# ============================================================================
# Best Practices:
# ============================================================================
#
# 1. **Use Unique Task IDs**
#    # ✅ Good - unique IDs
#    task = TaskWrapper(f"processor_{data_id}", function)
#
#    # ❌ Bad - duplicate IDs
#    task = TaskWrapper("processor", function)  # Reused ID!
#
# 2. **Set max_steps**
#    ctx.execute("start", max_steps=100)
#    - Prevents infinite iteration
#    - Safety limit for next_iteration()
#
# 3. **Use Channels for State**
#    channel = context.get_channel()
#    state = channel.get("state", default=initial_state)
#    channel.set("state", new_state)
#    - Persist state across iterations
#    - Share data between dynamic tasks
#
# 4. **Capture Variables in Lambdas**
#    # ✅ Good - capture with default argument
#    lambda v=value: process(v)
#
#    # ❌ Bad - closure over loop variable
#    lambda: process(value)  # May capture wrong value!
#
# 5. **Handle Errors in Dynamic Tasks**
#    try:
#        risky_operation()
#    except Exception as e:
#        error_task = TaskWrapper("error_handler", lambda: handle(e))
#        context.next_task(error_task)
#
# ============================================================================
# Advanced Patterns:
# ============================================================================
#
# **Goto Pattern (Jump to Existing Task)**:
# @task(inject_context=True)
# def controller(context):
#     if emergency_condition():
#         # Jump to existing task, skip successors
#         emergency_task = graph.get_node("emergency_handler")
#         context.next_task(emergency_task, goto=True)
#     else:
#         # Normal processing
#         context.next_task(normal_task)
#
# **Fork-Join Pattern**:
# @task(inject_context=True)
# def fork_tasks(context):
#     for item in items:
#         task = TaskWrapper(f"process_{item}", lambda: process(item))
#         context.next_task(task)
#     # All tasks will run, then join_task
#     join_task = TaskWrapper("join", aggregate_results)
#     context.next_task(join_task)
#
# **Recursive Processing**:
# @task(inject_context=True)
# def recursive_process(context):
#     depth = context.get_channel().get("depth", default=0)
#     if depth < max_depth:
#         context.get_channel().set("depth", depth + 1)
#         context.next_iteration()
#     else:
#         # Base case
#         return result
#
# ============================================================================
# Debugging Tips:
# ============================================================================
#
# 1. **Track Dynamic Tasks**
#    task_id = context.next_task(task)
#    print(f"Created task: {task_id}")
#
# 2. **Limit Iterations**
#    iteration = context.get_channel().get("iteration", default=0)
#    if iteration < 100:  # Safety limit
#        context.next_iteration()
#
# 3. **Log State Transitions**
#    print(f"State: {old_state} → {new_state}")
#
# 4. **Use max_steps**
#    ctx.execute("start", max_steps=50)  # Prevent runaway execution
#
# ============================================================================
