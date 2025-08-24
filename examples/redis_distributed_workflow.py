"""
Comprehensive Redis-based distributed workflow example.

This example demonstrates:
- Redis task queue for distributed task processing
- Redis channels for inter-task communication
- Redis coordinator for barrier synchronization
- GroupExecutor used indirectly through | operator
- Docker Redis container management
- Strongly typed task parameters and return values using type hints
- TypedChannel for type-safe communication

Usage:
    python examples/redis_distributed_workflow.py
"""

import contextlib
import random
import time
from typing import Optional, TypedDict

from graflow.channels.redis import RedisChannel
from graflow.channels.schemas import DataTransferMessage
from graflow.channels.typed import TypedChannel
from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.queue.factory import QueueBackend
from graflow.queue.redis import RedisTaskQueue
from graflow.worker.handler import InProcessTaskExecutor
from graflow.worker.worker import TaskWorker

try:
    import docker
    import redis
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install redis docker")
    DEPENDENCIES_AVAILABLE = False
    docker = None
    redis = None

# Strongly typed data structures using TypedDict
class ExtractionResult(TypedDict):
    """Result from data extraction task."""
    source: str
    records_count: int
    extraction_time: float
    timestamp: float


class TransformationResult(TypedDict):
    """Result from data transformation task."""
    source: str
    original_records: int
    processed_records: int
    transformation_time: float
    timestamp: float


class AggregationResult(TypedDict):
    """Result from data aggregation task."""
    total_records: int
    sources_processed: int
    aggregation_timestamp: float
    source_details: dict[str, TransformationResult]


class QualityCheckResult(TypedDict):
    """Result from quality check task."""
    total_records_checked: int
    data_quality_score: float
    sources_validated: int
    quality_check_timestamp: float
    passed: bool


@contextlib.contextmanager
def redis_docker_container():
    """Start and manage Redis Docker container."""
    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Redis and Docker libraries required")
    assert docker is not None, "Docker library is required for this example"
    assert redis is not None, "Redis library is required for this example"

    container = None
    try:
        print("🐳 Starting Redis Docker container...")
        client = docker.from_env()

        # Check if Redis container is already running
        try:
            existing_containers = client.containers.list(filters={"name": "graflow-redis"})
            if existing_containers:
                print("♻️  Using existing Redis container")
                container = existing_containers[0]
            else:
                container = client.containers.run(
                    "redis:7.2",
                    name="graflow-redis",
                    ports={'6379/tcp': 6379},
                    detach=True,
                    remove=True
                )
                print("⏳ Waiting for Redis to be ready...")
                time.sleep(3)
        except Exception as e:
            print(f"⚠️  Docker error: {e}")
            raise

        # Verify Redis connection
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        for attempt in range(5):
            try:
                redis_client.ping()
                print("✅ Redis connection established")
                break
            except redis.ConnectionError:
                if attempt < 4:
                    print(f"⏳ Waiting for Redis... (attempt {attempt + 1}/5)")
                    time.sleep(2)
                else:
                    raise

        yield redis_client

    finally:
        if container:
            try:
                print("🧹 Stopping Redis container...")
                container.stop()
            except Exception as e:
                print(f"⚠️  Error stopping container: {e}")



@task("extract_database", inject_context=True)
def extract_database_data(ctx: TaskExecutionContext) -> ExtractionResult:
    """Extract data from database source."""
    source = "database"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data extraction from {source}")

    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    time.sleep(processing_time)

    # Generate mock data
    result: ExtractionResult = {
        "source": source,
        "records_count": random.randint(100, 1000),
        "extraction_time": processing_time,
        "timestamp": time.time()
    }

    # Get channel from execution context
    channel = ctx.get_channel()
    channel.set(f"extracted_data_{source}", result)

    # Send typed message via typed channel
    typed_channel = ctx.get_typed_channel(DataTransferMessage)
    transfer_msg: DataTransferMessage = {
        "from_task": task_id,
        "to_task": f"transform_{source}",
        "data": result,
        "data_type": "extraction_result",
        "timestamp": time.time()
    }
    typed_channel.send(f"transfer_{task_id}", transfer_msg)

    print(f"[{task_id}] Completed extraction from {source}: {result['records_count']} records")
    return result


@task("extract_api", inject_context=True)
def extract_api_data(ctx: TaskExecutionContext) -> ExtractionResult:
    """Extract data from API source."""
    source = "api"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data extraction from {source}")

    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    time.sleep(processing_time)

    # Generate mock data
    result: ExtractionResult = {
        "source": source,
        "records_count": random.randint(100, 1000),
        "extraction_time": processing_time,
        "timestamp": time.time()
    }

    # Get channel from execution context
    channel = ctx.get_channel()
    channel.set(f"extracted_data_{source}", result)

    # Send typed message via typed channel
    typed_channel = ctx.get_typed_channel(DataTransferMessage)
    transfer_msg: DataTransferMessage = {
        "from_task": task_id,
        "to_task": f"transform_{source}",
        "data": result,
        "data_type": "extraction_result",
        "timestamp": time.time()
    }
    typed_channel.send(f"transfer_{task_id}", transfer_msg)

    print(f"[{task_id}] Completed extraction from {source}: {result['records_count']} records")
    return result


@task("extract_files", inject_context=True)
def extract_files_data(ctx: TaskExecutionContext) -> ExtractionResult:
    """Extract data from files source."""
    source = "files"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data extraction from {source}")

    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    time.sleep(processing_time)

    # Generate mock data
    result: ExtractionResult = {
        "source": source,
        "records_count": random.randint(100, 1000),
        "extraction_time": processing_time,
        "timestamp": time.time()
    }

    # Get channel from execution context
    channel = ctx.get_channel()
    channel.set(f"extracted_data_{source}", result)

    # Send typed message via typed channel
    typed_channel = ctx.get_typed_channel(DataTransferMessage)
    transfer_msg: DataTransferMessage = {
        "from_task": task_id,
        "to_task": f"transform_{source}",
        "data": result,
        "data_type": "extraction_result",
        "timestamp": time.time()
    }
    typed_channel.send(f"transfer_{task_id}", transfer_msg)

    print(f"[{task_id}] Completed extraction from {source}: {result['records_count']} records")
    return result


@task("extract_stream", inject_context=True)
def extract_stream_data(ctx: TaskExecutionContext) -> ExtractionResult:
    """Extract data from stream source."""
    source = "stream"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data extraction from {source}")

    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    time.sleep(processing_time)

    # Generate mock data
    result: ExtractionResult = {
        "source": source,
        "records_count": random.randint(100, 1000),
        "extraction_time": processing_time,
        "timestamp": time.time()
    }

    # Get channel from execution context
    channel = ctx.get_channel()
    channel.set(f"extracted_data_{source}", result)

    # Send typed message via typed channel
    typed_channel = ctx.get_typed_channel(DataTransferMessage)
    transfer_msg: DataTransferMessage = {
        "from_task": task_id,
        "to_task": f"transform_{source}",
        "data": result,
        "data_type": "extraction_result",
        "timestamp": time.time()
    }
    typed_channel.send(f"transfer_{task_id}", transfer_msg)

    print(f"[{task_id}] Completed extraction from {source}: {result['records_count']} records")
    return result


@task("transform_database", inject_context=True)
def transform_database_data(ctx: TaskExecutionContext) -> TransformationResult:
    """Transform extracted database data."""
    source = "database"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data transformation for {source}")

    channel = ctx.get_channel()

    # Wait for extraction data
    max_retries = 10
    extracted_data: Optional[ExtractionResult] = None
    for i in range(max_retries):
        extracted_data = channel.get(f"extracted_data_{source}")
        if extracted_data:
            break
        print(f"[{task_id}] Waiting for extraction data from {source} (attempt {i+1})")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"No extraction data found for {source}")

    # Simulate transformation
    transform_time = random.uniform(0.5, 1.5)
    time.sleep(transform_time)

    result: TransformationResult = {
        "source": source,
        "original_records": extracted_data["records_count"],
        "processed_records": int(extracted_data["records_count"] * 0.95),  # 95% success rate
        "transformation_time": transform_time,
        "timestamp": time.time()
    }

    # Store transformed data
    channel.set(f"transformed_data_{source}", result)
    print(f"[{task_id}] Completed transformation for {source}: {result['processed_records']} records")

    return result


@task("transform_api", inject_context=True)
def transform_api_data(ctx: TaskExecutionContext) -> TransformationResult:
    """Transform extracted API data."""
    source = "api"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data transformation for {source}")

    channel = ctx.get_channel()

    # Wait for extraction data
    max_retries = 10
    extracted_data: Optional[ExtractionResult] = None
    for i in range(max_retries):
        extracted_data = channel.get(f"extracted_data_{source}")
        if extracted_data:
            break
        print(f"[{task_id}] Waiting for extraction data from {source} (attempt {i+1})")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"No extraction data found for {source}")

    # Simulate transformation
    transform_time = random.uniform(0.5, 1.5)
    time.sleep(transform_time)

    result: TransformationResult = {
        "source": source,
        "original_records": extracted_data["records_count"],
        "processed_records": int(extracted_data["records_count"] * 0.95),  # 95% success rate
        "transformation_time": transform_time,
        "timestamp": time.time()
    }

    # Store transformed data
    channel.set(f"transformed_data_{source}", result)
    print(f"[{task_id}] Completed transformation for {source}: {result['processed_records']} records")

    return result


@task("transform_files", inject_context=True)
def transform_files_data(ctx: TaskExecutionContext) -> TransformationResult:
    """Transform extracted files data."""
    source = "files"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data transformation for {source}")

    channel = ctx.get_channel()

    # Wait for extraction data
    max_retries = 10
    extracted_data: Optional[ExtractionResult] = None
    for i in range(max_retries):
        extracted_data = channel.get(f"extracted_data_{source}")
        if extracted_data:
            break
        print(f"[{task_id}] Waiting for extraction data from {source} (attempt {i+1})")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"No extraction data found for {source}")

    # Simulate transformation
    transform_time = random.uniform(0.5, 1.5)
    time.sleep(transform_time)

    result: TransformationResult = {
        "source": source,
        "original_records": extracted_data["records_count"],
        "processed_records": int(extracted_data["records_count"] * 0.95),  # 95% success rate
        "transformation_time": transform_time,
        "timestamp": time.time()
    }

    # Store transformed data
    channel.set(f"transformed_data_{source}", result)
    print(f"[{task_id}] Completed transformation for {source}: {result['processed_records']} records")

    return result


@task("transform_stream", inject_context=True)
def transform_stream_data(ctx: TaskExecutionContext) -> TransformationResult:
    """Transform extracted stream data."""
    source = "stream"
    task_id = ctx.task_id
    print(f"[{task_id}] Starting data transformation for {source}")

    channel = ctx.get_channel()

    # Wait for extraction data
    max_retries = 10
    extracted_data: Optional[ExtractionResult] = None
    for i in range(max_retries):
        extracted_data = channel.get(f"extracted_data_{source}")
        if extracted_data:
            break
        print(f"[{task_id}] Waiting for extraction data from {source} (attempt {i+1})")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"No extraction data found for {source}")

    # Simulate transformation
    transform_time = random.uniform(0.5, 1.5)
    time.sleep(transform_time)

    result: TransformationResult = {
        "source": source,
        "original_records": extracted_data["records_count"],
        "processed_records": int(extracted_data["records_count"] * 0.95),  # 95% success rate
        "transformation_time": transform_time,
        "timestamp": time.time()
    }

    # Store transformed data
    channel.set(f"transformed_data_{source}", result)
    print(f"[{task_id}] Completed transformation for {source}: {result['processed_records']} records")

    return result


@task("aggregate_data", inject_context=True)
def aggregate_all_data(ctx: TaskExecutionContext) -> AggregationResult:
    """Aggregate all transformed data."""
    task_id = ctx.task_id
    sources = ["database", "api", "files", "stream"]
    print(f"[{task_id}] Starting data aggregation for sources: {sources}")

    channel = ctx.get_channel()

    result: AggregationResult = {
        "total_records": 0,
        "sources_processed": 0,
        "aggregation_timestamp": time.time(),
        "source_details": {}
    }

    # Wait for all transformation data
    for source in sources:
        max_retries = 15
        transformed_data: Optional[TransformationResult] = None
        for i in range(max_retries):
            transformed_data = channel.get(f"transformed_data_{source}")
            if transformed_data:
                result["total_records"] += transformed_data["processed_records"]
                result["sources_processed"] += 1
                result["source_details"][source] = transformed_data
                break
            print(f"[{task_id}] Waiting for transformation data from {source} (attempt {i+1})")
            time.sleep(0.5)
        else:
            print(f"[{task_id}] Warning: No transformation data found for {source}")

    # Store final results
    channel.set("final_aggregated_data", result)
    print(f"[{task_id}] Aggregation complete: {result['total_records']} total records from {result['sources_processed']} sources")

    return result


@task("quality_check", inject_context=True)
def perform_quality_check(ctx: TaskExecutionContext) -> QualityCheckResult:
    """Perform quality checks on aggregated data."""
    task_id = ctx.task_id
    print(f"[{task_id}] Starting quality check")

    channel = ctx.get_channel()

    # Wait for aggregated data
    max_retries = 20
    aggregated_data: Optional[AggregationResult] = None
    for i in range(max_retries):
        aggregated_data = channel.get("final_aggregated_data")
        if aggregated_data:
            break
        print(f"[{task_id}] Waiting for aggregated data (attempt {i+1})")
        time.sleep(0.5)
    else:
        raise TimeoutError("No aggregated data found for quality check")

    # Simulate quality check
    time.sleep(1.0)

    result: QualityCheckResult = {
        "total_records_checked": aggregated_data["total_records"],
        "data_quality_score": random.uniform(0.85, 0.98),
        "sources_validated": len(aggregated_data["source_details"]),
        "quality_check_timestamp": time.time(),
        "passed": True
    }

    # Store quality results
    channel.set("quality_check_results", result)
    print(f"[{task_id}] Quality check complete: Score {result['data_quality_score']:.3f}")

    return result


def create_task_registry():
    """Create task registry mapping task IDs to actual task functions."""
    return {
        "extract_database": extract_database_data,
        "extract_api": extract_api_data,
        "extract_files": extract_files_data,
        "extract_stream": extract_stream_data,
        "transform_database": transform_database_data,
        "transform_api": transform_api_data,
        "transform_files": transform_files_data,
        "transform_stream": transform_stream_data,
        "aggregate_data": aggregate_all_data,
        "quality_check": perform_quality_check
    }


class WorkflowTaskHandler(InProcessTaskExecutor):
    """Custom task handler that can execute workflow tasks."""

    def __init__(self, task_registry: dict, execution_context: ExecutionContext):
        """Initialize with task registry and execution context.

        Args:
            task_registry: Mapping of task_id to task functions
            execution_context: ExecutionContext for task execution
        """
        self.task_registry = task_registry
        self.execution_context = execution_context

    def _process_task(self, task) -> bool:
        """Execute task from TaskSpec using registered functions.

        Args:
            task: TaskSpec object from RedisTaskQueue

        Returns:
            bool: True if task completed successfully, False otherwise
        """
        task_id = task.task_id

        if task_id not in self.task_registry:
            raise ValueError(f"Task {task_id} not found in task registry")

        task_func = self.task_registry[task_id]

        try:
            # Create task execution context
            task_ctx = self.execution_context.create_task_context(task_id)
            self.execution_context.push_task_context(task_ctx)

            try:
                # Execute task with context injection if needed
                if hasattr(task_func, 'inject_context') and task_func.inject_context:
                    result = task_func(task_ctx)
                else:
                    result = task_func()

                # Store result in execution context
                self.execution_context.set_result(task_id, result)
                return True

            finally:
                self.execution_context.pop_task_context()

        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main workflow execution with TaskWorker."""
    if not DEPENDENCIES_AVAILABLE:
        return

    print("=== Redis Distributed Workflow Demo with TaskWorker ===\n")
    workers = []

    try:
        with redis_docker_container() as redis_client:
            # Initialize Redis channel for execution context
            redis_channel = RedisChannel("workflow_demo", redis_client=redis_client)
            redis_channel.clear()

            # We'll let the workflow create its own ExecutionContext,
            # but we'll provide the Redis configuration for later use

            # Create task registry for use by workers later
            task_registry = create_task_registry()

            # We'll set up workers after the workflow creates the execution context
            print("🔄 TaskWorkers will be set up after workflow execution context is ready...")

            # Configure GroupExecutor with Redis backend (now fixed)
            group_executor = GroupExecutor(
                backend=CoordinationBackend.REDIS,
                backend_config={
                    "redis_client": redis_client,
                    "key_prefix": "workflow_coordinator"
                }
            )

            # Create workflow context and execute
            with workflow("redis_distributed_processing") as wf:
                # Set Redis client and group executor
                wf.set_redis_client(redis_client)
                wf.set_group_executor(group_executor)

                print("\n🔧 Building workflow graph using >> and | operators...")

                # Define workflow using >> and | operators
                # Phase 1: Parallel extraction
                extraction_group = (extract_database_data | extract_api_data |
                                  extract_files_data | extract_stream_data)

                # Phase 2: Parallel transformation (depends on extraction)
                transformation_group = (transform_database_data | transform_api_data |
                                      transform_files_data | transform_stream_data)

                # Create workflow dependencies: extraction >> transformation >> aggregation >> quality_check
                extraction_group >> transformation_group >> aggregate_all_data >> perform_quality_check # type: ignore

                print("✅ Workflow graph built successfully")
                print("📊 Workflow Information:")
                wf.show_info()
                print()

                print("🚀 Executing distributed workflow with TaskWorkers...")

                # Use custom execution with Redis backend
                from graflow.core.context import ExecutionContext
                from graflow.core.engine import WorkflowEngine

                # Find start nodes
                start_nodes = wf.graph.get_start_nodes()
                if not start_nodes:
                    raise ValueError("No start nodes found in workflow")

                # Create execution context with Redis backend
                exec_context = ExecutionContext.create(
                    graph=wf.graph,
                    start_node=start_nodes[0],
                    max_steps=50,  # Allow more steps for distributed processing
                    queue_backend=QueueBackend.REDIS,
                    queue_config={
                        "redis_client": redis_client,
                        "key_prefix": "workflow_demo"
                    }
                )

                # Note: ExecutionContext uses its own channel for inter-task communication
                # The Redis channel will be used separately for workflow coordination

                if wf._group_executor:
                    exec_context.group_executor = wf._group_executor

                engine = WorkflowEngine()
                engine.execute(exec_context)
                print("✅ Workflow execution completed\n")

                # Get the task queue from execution context for monitoring
                redis_task_queue = exec_context.task_queue
                assert isinstance(redis_task_queue, RedisTaskQueue), "Expected RedisTaskQueue"

                # Now start TaskWorker processes with the proper queue
                print("\n🔄 Starting TaskWorker processes...")
                task_handler = WorkflowTaskHandler(task_registry, exec_context)

                for i in range(3):  # Start 3 workers
                    worker = TaskWorker(
                        queue=redis_task_queue,
                        handler=task_handler,
                        worker_id=f"worker-{i+1}",
                        max_concurrent_tasks=2,
                        poll_interval=0.5
                    )
                    worker.start()
                    workers.append(worker)
                    print(f"✅ Started worker-{i+1}")

                time.sleep(1)  # Give workers time to start

                # Wait for all tasks to complete
                print("⏳ Waiting for all tasks to complete...")
                timeout = 60  # 60 second timeout
                start_time = time.time()

                while time.time() - start_time < timeout:
                    if redis_task_queue.is_empty():
                        print("✅ All tasks completed")
                        break
                    time.sleep(1)
                    remaining = redis_task_queue.size()
                    if remaining > 0:
                        print(f"⏳ {remaining} tasks remaining...")
                else:
                    print("⚠️  Timeout waiting for tasks to complete")

                # Final Results
                print("\n📋 Final Results:")
                # Get results from execution context's channel
                execution_channel = exec_context.get_channel()
                final_results = execution_channel.get("quality_check_results")
                if final_results and isinstance(final_results, dict):
                    print(f"  - Total records processed: {final_results['total_records_checked']}")
                    print(f"  - Data quality score: {final_results['data_quality_score']:.3f}")
                    print(f"  - Sources validated: {final_results['sources_validated']}")
                    print(f"  - Quality check passed: {final_results['passed']}")

                print("\n🎉 Distributed workflow completed successfully!")

                # Show worker metrics if workers were started
                if workers:
                    print("\n📊 Worker Metrics:")
                    for worker in workers:
                        metrics = worker.get_metrics()
                        print(f"  {metrics['worker_id']}: "
                              f"Processed={metrics['tasks_processed']}, "
                              f"Success={metrics['tasks_succeeded']}, "
                              f"Failed={metrics['tasks_failed']}, "
                              f"Success Rate={metrics['success_rate']:.2%}")

            # Show typed channel usage
            print("\n📡 TypedChannel Messages:")
            if 'exec_context' in locals():
                execution_channel = exec_context.get_channel()
                typed_channel = TypedChannel(execution_channel, DataTransferMessage)
                transfer_keys = [key for key in typed_channel.keys() if key.startswith("transfer_")]
                for key in transfer_keys[:2]:  # Show first 2 messages
                    msg = typed_channel.receive(key)
                    if msg:
                        print(f"  {key}: {msg['from_task']} -> {msg['to_task']} ({msg['data_type']})")
            else:
                print("  No execution context available for typed channel messages")

            # Cleanup
            print("\n🧹 Cleaning up Redis data...")
            redis_channel.clear()
            if 'redis_task_queue' in locals():
                redis_task_queue.cleanup()
            redis_channel.close()

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop all workers
        print("\n🛑 Stopping TaskWorkers...")
        for worker in workers:
            if worker.is_running:
                worker.stop(timeout=10)
                print(f"✅ Stopped {worker.worker_id}")


if __name__ == "__main__":
    main()
