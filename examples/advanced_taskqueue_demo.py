"""Advanced TaskQueue features demonstration."""

from typing import cast

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.queue.base import TaskSpec
from graflow.queue.factory import QueueBackend


def demo_retry_functionality():
    """Demonstrate retry functionality."""
    print("=== Retry Functionality Demo ===")

    graph = TaskGraph()
    context = ExecutionContext(graph, start_node="retry_task")

    # Configure queue for retry
    context.task_queue.configure(enable_retry=True, enable_metrics=True)

    print(f"Queue configured - Retry: {context.task_queue.enable_retry}, Metrics: {context.task_queue.enable_metrics}")

    # Get initial task
    task_spec = context.task_queue.dequeue()
    assert task_spec is not None, "No task to retry"
    print(f"Initial task: {task_spec.task_id}, Retry count: {task_spec.retry_count}")

    # Simulate task failures
    for attempt in range(1, 5):
        print(f"\nAttempt {attempt}:")
        print(f"  Can retry: {task_spec.can_retry()}")

        if task_spec.can_retry():
            error_msg = f"Simulated error #{attempt}"
            should_retry = context.task_queue.handle_task_failure(task_spec, error_msg)

            print(f"  Failed with: {error_msg}")
            print(f"  Should retry: {should_retry}")
            print(f"  Retry count: {task_spec.retry_count}")
            print(f"  Status: {task_spec.status.value}")

            if should_retry:
                # Re-enqueue for retry
                context.task_queue.enqueue(task_spec)
                print("  Re-enqueued for retry")
        else:
            print("  Max retries exceeded!")
            break

    # Show final metrics
    metrics = context.task_queue.get_metrics()
    print(f"\nFinal metrics: {metrics}")
    print()


def demo_metrics_collection():
    """Demonstrate metrics collection."""
    print("=== Metrics Collection Demo ===")

    graph = TaskGraph()
    context = ExecutionContext(graph)

    # Configure metrics
    context.task_queue.configure(enable_metrics=True)
    print("Metrics enabled")

    # Initial metrics
    print(f"Initial metrics: {context.task_queue.get_metrics()}")

    # Simulate activity
    print("\nSimulating queue activity...")
    for i in range(3):
        context.add_to_queue(f"task_{i}")

    print(f"After enqueueing: {context.task_queue.get_metrics()}")

    # Process tasks
    while not context.is_completed():
        task = context.get_next_node()
        if task:
            print(f"  Processed: {task}")
        context.increment_step()

    print(f"After processing: {context.task_queue.get_metrics()}")

    # Reset metrics
    context.task_queue.reset_metrics()
    print(f"After reset: {context.task_queue.get_metrics()}")
    print()


def demo_task_spec_advanced_features():
    """Demonstrate TaskSpec advanced features."""
    print("=== TaskSpec Advanced Features Demo ===")

    graph = TaskGraph()
    context = ExecutionContext(graph)

    # Create task with custom retry settings
    task_spec = TaskSpec(
        task_id="advanced_task",
        execution_context=context,
        max_retries=2
    )

    print(f"Task: {task_spec.task_id}")
    print(f"Initial retry count: {task_spec.retry_count}")
    print(f"Max retries: {task_spec.max_retries}")
    print(f"Can retry: {task_spec.can_retry()}")

    # Simulate retries
    print("\nSimulating failures:")
    errors = ["Connection timeout", "Service unavailable", "Rate limit exceeded"]

    for _i, error in enumerate(errors, 1):
        if task_spec.can_retry():
            task_spec.increment_retry(error)
            print(f"  Retry {task_spec.retry_count}: {error}")
            print(f"    Status: {task_spec.status.value}")
            print(f"    Can retry: {task_spec.can_retry()}")
        else:
            print(f"  Cannot retry - max retries ({task_spec.max_retries}) exceeded")
            break

    print("\nFinal state:")
    print(f"  Last error: {task_spec.last_error}")
    print(f"  Total attempts: {task_spec.retry_count + 1}")
    print()


def demo_redis_advanced_features():
    """Demonstrate Redis advanced features."""
    print("=== Redis Advanced Features Demo ===")

    try:
        graph = TaskGraph()
        context = ExecutionContext(
            graph,
            start_node="redis_advanced_task",
            queue_backend=QueueBackend.REDIS,
            queue_config={
                'key_prefix': 'advanced_demo'
            }
        )

        # Configure advanced features
        from graflow.queue.redis import RedisTaskQueue
        redis_queue = cast(RedisTaskQueue, context.task_queue)
        redis_queue.configure(enable_retry=True, enable_metrics=True)

        print("Redis queue configured:")
        print(f"  Key prefix: {redis_queue.key_prefix}")
        print(f"  Session ID: {context.session_id}")
        print(f"  Retry enabled: {context.task_queue.enable_retry}")
        print(f"  Metrics enabled: {context.task_queue.enable_metrics}")

        # Add tasks with different retry settings
        for i in range(3):
            task_spec = TaskSpec(
                task_id=f"redis_task_{i}",
                execution_context=context,
                max_retries=i + 1  # Different retry limits
            )
            context.task_queue.enqueue(task_spec)

        print(f"\nAdded tasks, queue size: {context.task_queue.size()}")
        print(f"Metrics: {context.task_queue.get_metrics()}")

        # Process and simulate failures
        while not context.task_queue.is_empty():
            task_spec = context.task_queue.dequeue()
            if task_spec:
                print(f"\nProcessing: {task_spec.task_id}")
                print(f"  Max retries: {task_spec.max_retries}")

                # Simulate failure for demonstration
                if task_spec.retry_count == 0:  # First attempt
                    should_retry = context.task_queue.handle_task_failure(
                        task_spec, f"Simulated error for {task_spec.task_id}"
                    )
                    if should_retry:
                        context.task_queue.enqueue(task_spec)
                        print("  Re-enqueued for retry")

        print(f"\nFinal metrics: {context.task_queue.get_metrics()}")

        # Cleanup
        context.task_queue.cleanup()
        print("Redis cleanup completed")

    except ImportError:
        print("Redis not available")
    except Exception as e:
        print(f"Redis error: {e}")

    print()


def demo_configuration_driven_features():
    """Demonstrate configuration-driven advanced features."""
    print("=== Configuration-Driven Features Demo ===")

    configs = [
        {
            'name': 'Basic Queue',
            'config': {}
        },
        {
            'name': 'Retry Enabled',
            'config': {
                'enable_retry': True
            }
        },
        {
            'name': 'Full Featured',
            'config': {
                'enable_retry': True,
                'enable_metrics': True
            }
        }
    ]

    for config_info in configs:
        print(f"{config_info['name']}:")

        graph = TaskGraph()
        context = ExecutionContext(graph, start_node="config_test")

        # Apply configuration
        if config_info['config']:
            context.task_queue.configure(**config_info['config'])

        print(f"  Retry: {context.task_queue.enable_retry}")
        print(f"  Metrics: {context.task_queue.enable_metrics}")

        # Test task with failure
        task_spec = context.task_queue.dequeue()
        if task_spec:
            should_retry = context.task_queue.handle_task_failure(
                task_spec, "Test error"
            )
            print(f"  Task failed, should retry: {should_retry}")

            if context.task_queue.enable_metrics:
                print(f"  Metrics: {context.task_queue.get_metrics()}")

        print()


if __name__ == "__main__":
    demo_retry_functionality()
    demo_metrics_collection()
    demo_task_spec_advanced_features()
    demo_redis_advanced_features()
    demo_configuration_driven_features()

    print("🎉 Advanced TaskQueue features demonstration completed!")
