#!/usr/bin/env python3
"""
Simple Dynamic Task Generation Example

Shows basic usage of context.next_task() and context.next_iteration()
"""


from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


def demo_next_task():
    """Demonstrate basic next_task() functionality."""
    print("=== Demo: next_task() ===")

    # Create execution context with a dummy task
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Create a dummy task context to simulate being in a task
    dummy_task_ctx = context.create_task_context("start")
    context.push_task_context(dummy_task_ctx)

    # Create and add a dynamic task
    def dynamic_work():
        print("✨ Dynamic task executed!")
        return {"result": "dynamic_complete"}

    dynamic_task = TaskWrapper("dynamic_processor", dynamic_work)
    task_id = context.next_task(dynamic_task)

    print(f"✅ Created dynamic task: {task_id}")
    print(f"✅ Task added to graph: {task_id in context.graph.nodes}")
    print(f"✅ Task added to queue: {task_id in context.queue}")


def demo_next_iteration():
    """Demonstrate basic next_iteration() functionality."""
    print("\n=== Demo: next_iteration() ===")

    # Create execution context with a base task
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Add a base task to the graph first
    def base_task(ctx, data):
        count = data.get("count", 0)
        print(f"🔄 Iteration {count}, data: {data}")
        return {"processed": True, "count": count}

    base_wrapper = TaskWrapper("counter", base_task)
    graph.add_node("counter", task=base_wrapper)

    # Set current task context and create iteration
    counter_task_ctx = context.create_task_context("counter")
    context.push_task_context(counter_task_ctx)

    iteration_id = context.next_iteration({"count": 1, "message": "hello"})

    print(f"✅ Created iteration task: {iteration_id}")
    print(f"✅ Iteration task in graph: {iteration_id in context.graph.nodes}")
    print(f"✅ Iteration task in queue: {iteration_id in context.queue}")


def demo_conditional_creation():
    """Demonstrate conditional task creation."""
    print("\n=== Demo: Conditional Task Creation ===")

    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Create a processor task context
    processor_task_ctx = context.create_task_context("processor")
    context.push_task_context(processor_task_ctx)

    def process_data(value):
        print(f"🔍 Processing value: {value}")

        if value > 50:  # noqa: PLR2004
            # High value processing
            high_task = TaskWrapper(
                "high_value_handler",
                lambda: print(f"📈 High value processing: {value}")
            )
            task_id = context.next_task(high_task)
            print(f"✨ Created high-value task: {task_id}")

        elif value > 10:  # noqa: PLR2004
            # Medium value processing
            med_task = TaskWrapper(
                "medium_value_handler",
                lambda: print(f"📊 Medium value processing: {value}")
            )
            task_id = context.next_task(med_task)
            print(f"✨ Created medium-value task: {task_id}")

        else:
            # Low value - just log
            print(f"📉 Low value, no additional processing needed: {value}")

        return {"processed": value}

    # Test with different values
    print("\n🧪 Testing with value 75:")
    process_data(75)

    print("\n🧪 Testing with value 25:")
    process_data(25)

    print("\n🧪 Testing with value 5:")
    process_data(5)


def demo_iteration_chain():
    """Demonstrate chained iterations."""
    print("\n=== Demo: Iteration Chain ===")

    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Create a counter task
    def counter_task(ctx, data):
        count = data.get("count", 0)
        limit = data.get("limit", 3)

        print(f"🔢 Count: {count}/{limit}")

        if count < limit:
            # Continue counting
            next_data = {"count": count + 1, "limit": limit}
            iteration_id = ctx.next_iteration(next_data)
            print(f"➡️  Scheduled next iteration: {iteration_id}")
        else:
            # Done - create completion task
            done_task = TaskWrapper(
                "completion_handler",
                lambda: print("🎉 Counting completed!")
            )
            task_id = ctx.next_task(done_task)
            print(f"✅ Created completion task: {task_id}")

        return {"count": count, "done": count >= limit}

    # Add counter to graph
    counter_wrapper = TaskWrapper("iteration_counter", counter_task)
    graph.add_node("iteration_counter", task=counter_wrapper)

    # Create iteration counter task context
    iteration_counter_ctx = context.create_task_context("iteration_counter")
    context.push_task_context(iteration_counter_ctx)

    # Start the chain
    print("🚀 Starting iteration chain:")
    counter_task(context, {"count": 0, "limit": 3})


def main():
    """Run all demos."""
    print("🎯 Dynamic Task Generation Examples")
    print("=" * 50)

    try:
        demo_next_task()
        demo_next_iteration()
        demo_conditional_creation()
        demo_iteration_chain()

        print("\n" + "=" * 50)
        print("🎉 All demos completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback  # noqa: PLC0415
        traceback.print_exc()


if __name__ == "__main__":
    main()
