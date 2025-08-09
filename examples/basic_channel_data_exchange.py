#!/usr/bin/env python3
"""
Basic Channel Data Exchange Example

This example demonstrates how to use get_channel() for inter-task
communication and data exchange between tasks.

Key differences from TypedChannel:
- Uses channel.set(key, data) and channel.get(key) methods
- No type validation - any data type can be stored/retrieved
- More flexible but less type-safe than TypedChannel
- Good for simple data exchange patterns

For type-safe communication with validation, see typed_channel_data_exchange.py
"""

import traceback

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def demo_simple_data_exchange():  # noqa: PLR0915
    """Demonstrate basic data exchange using get_channel()."""
    print("📡 Basic Channel Data Exchange Demo")
    print("=" * 50)

    with workflow("basic_data_exchange") as wf:

        @task(inject_context=True)
        def producer_task(context: TaskExecutionContext) -> None:
            """Producer task that sends data via channel."""
            print("🏭 Producer: Creating and sending data...")

            # Get the communication channel
            channel = context.get_channel()

            # Send various types of data
            data_items = [
                ("item_1", {"name": "Product A", "price": 100, "category": "electronics"}),
                ("item_2", {"name": "Product B", "price": 200, "category": "books"}),
                ("item_3", {"name": "Product C", "price": 150, "category": "electronics"}),
                ("config", {"max_items": 10, "filter_category": "electronics"}),
                ("raw_numbers", [1, 2, 3, 4, 5])
            ]

            for key, data in data_items:
                channel.set(key, data)
                print(f"  📤 Sent {key}: {data}")

        @task(inject_context=True)
        def filter_task(context: TaskExecutionContext) -> None:
            """Filter task that processes data from the channel."""
            print("\n🔍 Filter: Processing data from channel...")

            # Get the same communication channel
            channel = context.get_channel()

            # Get configuration
            config = channel.get("config")
            if not config:
                print("  ❌ No configuration found!")
                return

            filter_category = config.get("filter_category", "all")
            print(f"  ⚙️ Filtering for category: {filter_category}")

            # Process items
            filtered_items = []
            for key in channel.keys():
                if key.startswith("item_"):
                    item = channel.get(key)
                    if item and item.get("category") == filter_category:
                        filtered_items.append(item)
                        print(f"  ✅ Filtered item: {item['name']} (${item['price']})")

            # Send filtered results back to channel
            channel.set("filtered_results", filtered_items)

        @task(inject_context=True)
        def analyzer_task(context: TaskExecutionContext) -> None:
            """Analyzer task that computes statistics from filtered data."""
            print("\n📊 Analyzer: Computing statistics...")

            channel = context.get_channel()

            # Get filtered results
            filtered_items = channel.get("filtered_results")
            if not filtered_items:
                print("  ❌ No filtered results found!")
                return

            # Compute statistics
            total_price = sum(item["price"] for item in filtered_items)
            avg_price = total_price / len(filtered_items) if filtered_items else 0
            item_count = len(filtered_items)

            stats = {
                "total_items": item_count,
                "total_price": total_price,
                "average_price": avg_price,
                "categories": list(set(item["category"] for item in filtered_items))
            }

            # Send stats back to channel
            channel.set("statistics", stats)
            print(f"  📊 Statistics computed: {stats}")

        @task(inject_context=True)
        def reporter_task(context: TaskExecutionContext) -> None:
            """Reporter task that displays final results."""
            print("\n📋 Reporter: Generating final report...")

            channel = context.get_channel()

            # Get all data for the report
            filtered_items = channel.get("filtered_results")
            stats = channel.get("statistics")
            raw_numbers = channel.get("raw_numbers")

            print("\n" + "="*40)
            print("📈 FINAL REPORT")
            print("="*40)

            if stats:
                print("📊 Statistics:")
                print(f"  • Total Items: {stats['total_items']}")
                print(f"  • Total Price: ${stats['total_price']}")
                print(f"  • Average Price: ${stats['average_price']:.2f}")
                print(f"  • Categories: {', '.join(stats['categories'])}")

            if filtered_items:
                print("\n📦 Filtered Items:")
                for item in filtered_items:
                    print(f"  • {item['name']}: ${item['price']}")

            if raw_numbers:
                print(f"\n🔢 Raw Numbers Sum: {sum(raw_numbers)}")

            print("="*40)

        # Build workflow pipeline
        producer_task >> filter_task >> analyzer_task >> reporter_task # type: ignore

        # Execute workflow
        wf.execute("producer_task")


def demo_dynamic_communication():
    """Demonstrate dynamic communication patterns."""
    print("\n🔄 Dynamic Communication Demo")
    print("=" * 50)

    with workflow("dynamic_communication") as wf:

        @task(inject_context=True)
        def coordinator_task(context: TaskExecutionContext) -> None:
            """Coordinator that manages task assignments."""
            print("🎯 Coordinator: Managing task assignments...")

            channel = context.get_channel()

            # Create work assignments
            assignments = [
                {"task_id": "worker_1", "work_type": "process", "data": [1, 2, 3]},
                {"task_id": "worker_2", "work_type": "transform", "data": ["a", "b", "c"]},
                {"task_id": "worker_3", "work_type": "validate", "data": {"x": 10, "y": 20}}
            ]

            for assignment in assignments:
                channel.set(f"assignment_{assignment['task_id']}", assignment)
                print(f"  📋 Assigned work to {assignment['task_id']}: {assignment['work_type']}")

            # Send coordination metadata
            channel.set("coordination_info", {
                "total_assignments": len(assignments),
                "start_time": "2024-01-01T10:00:00Z"
            })

        @task(inject_context=True)
        def worker_task(context: TaskExecutionContext) -> None:
            """Generic worker that processes assignments."""
            print("\n👷 Worker: Processing assignments...")

            channel = context.get_channel()

            # Process all assignments
            results = []
            for key in channel.keys():
                if key.startswith("assignment_"):
                    assignment = channel.get(key)
                    if assignment:
                        work_type = assignment["work_type"]
                        data = assignment["data"]

                        # Simulate different types of work
                        if work_type == "process":
                            result = sum(data) if isinstance(data, list) else 0
                        elif work_type == "transform":
                            result = [item.upper() for item in data] if isinstance(data, list) else []
                        elif work_type == "validate":
                            result = all(isinstance(v, int | float) for v in data.values()) if isinstance(data, dict) else False  # noqa: E501
                        else:
                            result = f"Unknown work type: {work_type}"

                        result_data = {
                            "task_id": assignment["task_id"],
                            "work_type": work_type,
                            "original_data": data,
                            "result": result
                        }
                        results.append(result_data)
                        print(f"  ⚙️ Processed {assignment['task_id']}: {result}")

            # Send all results back
            channel.set("worker_results", results)

        @task(inject_context=True)
        def collector_task(context: TaskExecutionContext) -> None:
            """Collector that aggregates all results."""
            print("\n📦 Collector: Aggregating results...")

            channel = context.get_channel()

            # Get coordination info and results
            coordination_info = channel.get("coordination_info")
            worker_results = channel.get("worker_results")

            if coordination_info and worker_results:
                print("\n📊 Collection Summary:")
                print(f"  • Expected assignments: {coordination_info['total_assignments']}")
                print(f"  • Completed assignments: {len(worker_results)}")
                print(f"  • Start time: {coordination_info['start_time']}")

                print("\n📋 Results by task:")
                for result in worker_results:
                    print(f"  • {result['task_id']} ({result['work_type']}): {result['result']}")

        # Build workflow
        coordinator_task >> worker_task >> collector_task # type: ignore

        # Execute workflow
        wf.execute("coordinator_task")


def demo_error_handling():
    """Demonstrate error handling in channel communication."""
    print("\n🛡️ Error Handling Demo")
    print("=" * 50)

    with workflow("error_handling_demo") as wf:

        @task(inject_context=True)
        def sender_task(context: TaskExecutionContext) -> None:
            """Send data including some problematic cases."""
            print("📤 Sender: Sending various data types...")

            channel = context.get_channel()

            # Send various data types including edge cases
            test_data = [
                ("normal_data", {"status": "ok", "value": 42}),
                ("empty_data", {}),
                ("none_data", None),
                ("large_data", list(range(1000))),
                ("nested_data", {"level1": {"level2": {"level3": "deep"}}})
            ]

            for key, data in test_data:
                try:
                    channel.set(key, data)
                    print(f"  ✅ Sent {key}: {type(data).__name__}")
                except Exception as e:
                    print(f"  ❌ Failed to send {key}: {e}")

        @task(inject_context=True)
        def receiver_task(context: TaskExecutionContext) -> None:
            """Receive and validate data with error handling."""
            print("\n📥 Receiver: Processing received data...")

            channel = context.get_channel()

            # Try to receive and process all data
            for key in channel.keys():
                try:
                    data = channel.get(key)
                    if data is None:
                        print(f"  ⚠️ {key}: Received None")
                    elif isinstance(data, dict) and not data:
                        print(f"  ⚠️ {key}: Received empty dict")
                    elif isinstance(data, list) and len(data) > 100:
                        print(f"  📊 {key}: Large list with {len(data)} items")
                    else:
                        print(f"  ✅ {key}: {type(data).__name__} - {str(data)[:50]}...")
                except Exception as e:
                    print(f"  ❌ Error processing {key}: {e}")

        # Build workflow
        sender_task >> receiver_task # type: ignore

        # Execute workflow
        wf.execute("sender_task")


def main():
    """Run basic channel communication examples."""
    print("📡 Basic Channel Inter-Task Communication Examples")
    print("This demonstrates data exchange using get_channel()")

    try:
        # Simple data exchange demo
        demo_simple_data_exchange()

        # Dynamic communication demo
        demo_dynamic_communication()

        # Error handling demo
        demo_error_handling()

        print("\n🎉 All basic channel examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
