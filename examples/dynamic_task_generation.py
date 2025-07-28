#!/usr/bin/env python3
"""
Dynamic Task Generation Example for Graflow

This example demonstrates how to use next_task() and next_iteration()
for creating dynamic, data-driven workflows that generate tasks at runtime.
"""

import random
import traceback

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import TaskWrapper
from graflow.core.workflow import workflow


# Example 1: Conditional Dynamic Task Generation
@task(id="data_classifier", inject_context=True)
def classify_data(context: TaskExecutionContext):
    """Classify data and generate different processing tasks based on the classification."""
    data = context.get_local_data
    print(f"Classifying data: {data}")

    data_type = data.get("type", "unknown")
    value = data.get("value", 0)

    if data_type == "numerical" and value > 100:
        # Create high-value numerical processing task
        high_value_task = TaskWrapper(
            "high_value_processor",
            lambda: process_high_value_numerical(data)
        )
        task_id = context.next_task(high_value_task)
        print(f"Generated high-value processing task: {task_id}")

    elif data_type == "numerical":
        # Create standard numerical processing task
        standard_task = TaskWrapper(
            "standard_processor",
            lambda: process_standard_numerical(data)
        )
        task_id = context.next_task(standard_task)
        print(f"Generated standard processing task: {task_id}")

    elif data_type == "text":
        # Create text processing task
        text_task = TaskWrapper(
            "text_processor",
            lambda: process_text_data(data)
        )
        task_id = context.next_task(text_task)
        print(f"Generated text processing task: {task_id}")

    else:
        print(f"Unknown data type: {data_type}, skipping processing")

    return {"classified_as": data_type, "value": value}


def process_high_value_numerical(data):
    """Process high-value numerical data with special handling."""
    result = data["value"] * 1.5 + 50
    print(f"High-value processing: {data['value']} -> {result}")
    return {"processed_value": result, "processing_type": "high_value"}


def process_standard_numerical(data):
    """Process standard numerical data."""
    result = data["value"] * 1.2
    print(f"Standard processing: {data['value']} -> {result}")
    return {"processed_value": result, "processing_type": "standard"}


def process_text_data(data):
    """Process text data."""
    text = data.get("text", "")
    result = f"PROCESSED: {text.upper()}"
    print(f"Text processing: '{text}' -> '{result}'")
    return {"processed_text": result, "length": len(text)}


# Example 2: Iterative Processing with Dynamic Task Generation
@task(id="iterative_optimizer")
def optimize_parameters(context, params):
    """Optimize parameters iteratively until convergence."""
    print(f"Optimizing parameters: {params}")

    iteration = params.get("iteration", 0)
    accuracy = params.get("accuracy", 0.5)
    learning_rate = params.get("learning_rate", 0.1)

    # Simulate optimization step
    new_accuracy = min(accuracy + 0.15, 0.95)
    new_learning_rate = learning_rate * 0.95

    updated_params = {
        "iteration": iteration + 1,
        "accuracy": new_accuracy,
        "learning_rate": new_learning_rate
    }

    print(f"Iteration {iteration + 1}: accuracy={new_accuracy:.3f}, lr={new_learning_rate:.3f}")

    # Check convergence conditions
    if new_accuracy >= 0.9 or iteration >= 5:
        # Converged - create final result task
        final_task = TaskWrapper(
            "save_final_model",
            lambda: save_optimized_model(updated_params)
        )
        task_id = context.next_task(final_task)
        print(f"Convergence reached, creating final task: {task_id}")
    else:
        # Continue optimization - create next iteration
        print("Not converged yet, scheduling next iteration...")
        context.next_iteration(updated_params)

    return updated_params


def save_optimized_model(params):
    """Save the final optimized model."""
    print(f"Saving optimized model with accuracy: {params['accuracy']:.3f}")
    return {
        "model_saved": True,
        "final_accuracy": params["accuracy"],
        "total_iterations": params["iteration"]
    }


# Example 3: Parallel Dynamic Task Generation
@task(id="batch_coordinator")
def coordinate_batch_processing(context, batch):
    """Coordinate processing of a batch by creating parallel tasks."""
    print(f"Coordinating batch processing for {len(batch['items'])} items")

    # Create parallel processing tasks for each item
    for i, item in enumerate(batch["items"]):
        item_task = TaskWrapper(
            f"process_batch_item_{i}",
            lambda item=item, idx=i: process_batch_item(item, idx)
        )
        task_id = context.next_task(item_task)
        print(f"Created batch item task {i}: {task_id}")

    # Create aggregation task to collect results
    aggregator_task = TaskWrapper(
        "aggregate_batch_results",
        lambda: aggregate_batch_results(len(batch["items"]))
    )
    task_id = context.next_task(aggregator_task)
    print(f"Created aggregation task: {task_id}")

    return {"batch_size": len(batch["items"]), "tasks_created": len(batch["items"]) + 1}


def process_batch_item(item, index):
    """Process a single batch item."""
    processed = item * 2 + index
    print(f"Processed batch item {index}: {item} -> {processed}")
    return {"item_index": index, "original": item, "processed": processed}


def aggregate_batch_results(expected_count):
    """Aggregate results from batch processing."""
    print(f"Aggregating results from {expected_count} batch items")
    return {"aggregation_complete": True, "processed_count": expected_count}


# Example 4: Error Handling with Dynamic Recovery Tasks
@task(id="resilient_processor")
def process_with_error_handling(context: ExecutionContext, data):
    """Process data with dynamic error recovery."""
    print(f"Processing data: {data}")

    retry_count = data.get("retry_count", 0)
    max_retries = 3

    # Simulate processing that might fail
    success_rate = 0.4 + (retry_count * 0.2)  # Increase success rate with retries

    if random.random() < success_rate:
        # Success - create success handler task
        print("Processing successful!")
        success_task = TaskWrapper(
            "handle_success",
            lambda: handle_processing_success(data)
        )
        task_id = context.next_task(success_task)
        print(f"Created success handler: {task_id}")
        return {"status": "success", "result": data["value"] * 2}

    else:
        # Failure
        print(f"Processing failed (attempt {retry_count + 1})")

        if retry_count < max_retries:
            # Retry with updated data
            retry_data = data.copy()
            retry_data["retry_count"] = retry_count + 1
            print(f"Scheduling retry {retry_count + 1}")
            context.next_iteration(retry_data)
            return {"status": "retrying", "attempt": retry_count + 1}

        else:
            # Max retries reached - create error handler
            print("Max retries reached, creating error handler")
            error_task = TaskWrapper(
                "handle_error",
                lambda: handle_processing_error(data)
            )
            task_id = context.next_task(error_task)
            print(f"Created error handler: {task_id}")
            return {"status": "failed", "final_attempt": retry_count + 1}


def handle_processing_success(data):
    """Handle successful processing."""
    print(f"Successfully processed data: {data}")
    return {"success_handled": True, "data": data}


def handle_processing_error(data):
    """Handle processing error."""
    print(f"Handling error for data: {data}")
    return {"error_handled": True, "data": data, "logged": True}


def run_classification_example():
    """Run the conditional dynamic task generation example."""
    print("=" * 60)
    print("EXAMPLE 1: Conditional Dynamic Task Generation")
    print("=" * 60)

    with workflow("dynamic_classification") as wf:
        # Test with different data types
        test_data = [
            {"type": "numerical", "value": 150},    # High-value numerical
            {"type": "numerical", "value": 50},     # Standard numerical
            {"type": "text", "text": "hello world"}, # Text data
            {"type": "unknown", "value": 10}        # Unknown type
        ]

        for data in test_data:
            print(f"\n🚀 Classifying data: {data}")
            wf.execute("data_classifier", data=data)


def run_optimization_example():
    """Run the iterative optimization example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Iterative Optimization with Dynamic Tasks")
    print("=" * 60)

    with workflow("iterative_optimization") as wf:
        initial_params = {
            "accuracy": 0.3,
            "learning_rate": 0.1,
            "iteration": 0
        }
        optimize_parameters >> optimize_parameters(initial_params)
        wf.execute()


def run_batch_processing_example():
    """Run the parallel batch processing example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Parallel Batch Processing")
    print("=" * 60)

    with workflow("batch_processing") as wf:
        batch_data = {"items": [10, 20, 30, 40, 50]}
        coordinate_batch_processing >> coordinate_batch_processing(batch_data)
        wf.execute()


def run_error_handling_example():
    """Run the error handling with recovery example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Error Handling with Dynamic Recovery")
    print("=" * 60)

    with workflow("resilient_processing") as wf:
        test_data = {"value": 42, "id": "test_001"}
        process_with_error_handling >> process_with_error_handling(test_data)
        wf.execute()


def main():
    """Run all dynamic task generation examples."""
    print("Dynamic Task Generation Examples for Graflow")
    print("=" * 60)

    try:
        run_classification_example()
        run_optimization_example()
        run_batch_processing_example()
        run_error_handling_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()