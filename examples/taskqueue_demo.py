"""TaskQueue demonstration showing different backends."""

from typing import cast

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.queue.factory import QueueBackend, TaskQueueFactory


def demo_in_memory_backend():
    """Demonstrate in-memory TaskQueue backend."""
    print("=== In-Memory TaskQueue Demo ===")

    graph = TaskGraph()

    # Default backend (backwards compatible)
    context = ExecutionContext(graph, start_node="task1")
    print(f"Backend: {type(context.task_queue).__name__}")
    print(f"Initial queue size: {context.queue.size()}")

    # Add tasks
    context.add_to_queue("task2")
    context.add_to_queue("task3")
    print(f"After adding tasks - queue size: {context.queue.size()}")

    # Process tasks
    while not context.is_completed():
        next_task = context.get_next_node()
        print(f"Processing: {next_task}")
        context.increment_step()

    print(f"Final state - completed: {context.is_completed()}")
    print()


def demo_explicit_in_memory_backend():
    """Demonstrate explicit in-memory backend specification."""
    print("=== Explicit In-Memory Backend Demo ===")

    graph = TaskGraph()

    # Explicit backend specification
    context = ExecutionContext(
        graph,
        start_node="start",
        queue_backend=QueueBackend.IN_MEMORY
    )

    print(f"Backend: {type(context.task_queue).__name__}")
    print(f"Queue size: {context.task_queue.size()}")
    print(f"Next node (peek): {context.task_queue.peek_next_node()}")

    # Process first node
    node = context.get_next_node()
    print(f"Got node: {node}")
    print(f"Queue now empty: {context.is_completed()}")
    print()


def demo_redis_backend():
    """Demonstrate Redis TaskQueue backend."""
    print("=== Redis TaskQueue Demo ===")

    try:
        graph = TaskGraph()

        # Redis backend with configuration
        context = ExecutionContext(
            graph,
            start_node="redis_task1",
            queue_backend=QueueBackend.REDIS,
            queue_config={
                'key_prefix': 'demo',
                # 'redis_client': custom_redis_client  # Optional
            }
        )

        print(f"Backend: {type(context.task_queue).__name__}")
        from graflow.queue.redis import RedisTaskQueue  # noqa: PLC0415
        redis_queue = cast('RedisTaskQueue', context.task_queue)
        print(f"Redis key prefix: {redis_queue.key_prefix}")
        print(f"Session ID: {context.session_id}")

        # Add tasks
        context.add_to_queue("redis_task2")
        context.add_to_queue("redis_task3")

        print(f"Queue size: {context.task_queue.size()}")
        print(f"Is empty: {context.is_completed()}")

        # Process tasks
        tasks_processed = []
        while not context.is_completed() and context.steps < 5:
            next_task = context.get_next_node()
            if next_task:
                print(f"Processing: {next_task}")
                tasks_processed.append(next_task)
            context.increment_step()

        print(f"Processed tasks: {tasks_processed}")

        # Cleanup
        context.task_queue.cleanup()
        print("Redis keys cleaned up")

    except ImportError:
        print("Redis not available. Install with: pip install redis")
    except Exception as e:
        print(f"Redis connection error: {e}")
        print("Make sure Redis server is running on localhost:6379")

    print()


def demo_configuration_based():
    """Demonstrate configuration-based TaskQueue creation."""
    print("=== Configuration-Based Demo ===")

    # Configuration for different backends
    configs = [
        {
            'name': 'In-Memory Config',
            'config': {
                'backend': 'in_memory',
                'config': {}
            }
        },
        {
            'name': 'Redis Config',
            'config': {
                'backend': 'redis',
                'config': {
                    'key_prefix': 'config_demo'
                }
            }
        }
    ]

    for demo_config in configs:
        try:
            print(f"Testing {demo_config['name']}:")

            graph = TaskGraph()
            context = ExecutionContext(graph, start_node="config_task")

            # Create queue from configuration
            queue = TaskQueueFactory.create_from_config(
                context, demo_config['config']
            )

            print(f"  Backend: {type(queue).__name__}")
            print(f"  Initial size: {queue.size()}")

            queue.add_node("test")
            print(f"  After add: {queue.size()}")

            node = queue.get_next_node()
            print(f"  Got node: {node}")
            print(f"  Final size: {queue.size()}")

            # Cleanup if available
            if hasattr(queue, 'cleanup'):
                queue.cleanup()

        except Exception as e:
            print(f"  Error: {e}")

        print()


if __name__ == "__main__":
    demo_in_memory_backend()
    demo_explicit_in_memory_backend()
    demo_redis_backend()
    demo_configuration_based()

    print("🎉 TaskQueue demonstration completed!")
