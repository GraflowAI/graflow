"""Example demonstrating typed channel usage in graflow tasks."""

import time
from typing import TypedDict

from graflow.channels.schemas import TaskProgressMessage, TaskResultMessage
from graflow.core.context import TaskExecutionContext, create_execution_context
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


class ProcessingDataMessage(TypedDict):
    """Custom message type for data processing."""
    data: list[int]
    processing_type: str
    batch_id: str
    timestamp: float


@task("data_generator", inject_context=True)
def generate_data(ctx: TaskExecutionContext) -> list[int]:
    """Generate some data and notify via typed channel."""
    data = [1, 2, 3, 4, 5]

    # Get typed channel for progress updates
    progress_channel = ctx.get_typed_channel(TaskProgressMessage)

    # Send progress message
    progress_msg: TaskProgressMessage = {
        "task_id": ctx.task_id,
        "progress": 0.5,
        "message": "Generated data",
        "timestamp": time.time()
    }
    progress_channel.send("progress", progress_msg)

    # Get custom typed channel for data transfer
    data_channel = ctx.get_typed_channel(ProcessingDataMessage)

    # Send data to processor
    data_msg: ProcessingDataMessage = {
        "data": data,
        "processing_type": "sum",
        "batch_id": "batch_001",
        "timestamp": time.time()
    }
    data_channel.send("processing_data", data_msg)

    return data


@task("data_processor", inject_context=True)
def process_data(ctx: TaskExecutionContext) -> int:
    """Process data received via typed channel."""
    # Get custom typed channel
    data_channel = ctx.get_typed_channel(ProcessingDataMessage)

    # Receive data (type-safe)
    data_msg = data_channel.receive("processing_data")

    if data_msg is None:
        # Send error via typed channel
        result_channel = ctx.get_typed_channel(TaskResultMessage)
        error_msg: TaskResultMessage = {
            "task_id": ctx.task_id,
            "result": None,
            "timestamp": time.time(),
            "status": "error"
        }
        result_channel.send("result", error_msg)
        return 0

    # Process the data
    data = data_msg["data"]
    processing_type = data_msg["processing_type"]

    if processing_type == "sum":
        result = sum(data)
    elif processing_type == "product":
        result = 1
        for x in data:
            result *= x
    else:
        result = len(data)

    # Send result via typed channel
    result_channel = ctx.get_typed_channel(TaskResultMessage)
    result_msg: TaskResultMessage = {
        "task_id": ctx.task_id,
        "result": result,
        "timestamp": time.time(),
        "status": "completed"
    }
    result_channel.send("result", result_msg)

    # Send final progress
    progress_channel = ctx.get_typed_channel(TaskProgressMessage)
    progress_msg: TaskProgressMessage = {
        "task_id": ctx.task_id,
        "progress": 1.0,
        "message": f"Processed data, result: {result}",
        "timestamp": time.time()
    }
    progress_channel.send("progress", progress_msg)

    return result


@task("result_collector", inject_context=True)
def collect_results(ctx: TaskExecutionContext) -> dict:
    """Collect results from typed channels."""
    results = {}

    # Get result channel
    result_channel = ctx.get_typed_channel(TaskResultMessage)
    result_msg = result_channel.receive("result")

    if result_msg:
        results["task_result"] = result_msg

    # Get progress channel
    progress_channel = ctx.get_typed_channel(TaskProgressMessage)
    progress_msg = progress_channel.receive("progress")

    if progress_msg:
        results["progress"] = progress_msg

    return results


def main():
    """Run the typed channel example."""
    # Create task graph
    graph = TaskGraph()

    # Add tasks to graph
    graph.add_node(generate_data)
    graph.add_node(process_data)
    graph.add_node(collect_results)

    # Add dependencies
    graph.add_edge("data_generator", "data_processor")
    graph.add_edge("data_processor", "result_collector")

    # Create execution context and execute
    context = create_execution_context("data_generator", max_steps=5)
    context.graph = graph
    context.execute()

    print("Execution results:")
    for task_id in context.executed:
        result = context.get_result(task_id)
        if result is not None:
            print(f"  {task_id}: {result}")

    # Access typed channel data
    channel = context.get_channel()
    print(f"\nChannel keys: {channel.keys()}")

    # Try to access some messages
    if "result" in channel.keys():
        print(f"Result message: {channel.get('result')}")
    if "progress" in channel.keys():
        print(f"Progress message: {channel.get('progress')}")
    if "processing_data" in channel.keys():
        print(f"Processing data message: {channel.get('processing_data')}")


if __name__ == "__main__":
    main()
