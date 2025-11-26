"""Task worker implementation for processing tasks from queues."""

import logging
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from graflow.core.engine import WorkflowEngine
from graflow.exceptions import GraflowRuntimeError
from graflow.queue.base import TaskSpec
from graflow.queue.redis import DistributedTaskQueue

if TYPE_CHECKING:
    from graflow.trace.base import Tracer

logger = logging.getLogger(__name__)


class TaskWorker:
    """Worker that processes tasks from a queue using WorkflowEngine."""

    def __init__(self, queue: DistributedTaskQueue, worker_id: str,
                 max_concurrent_tasks: int = 4, poll_interval: float = 0.1,
                 graceful_shutdown_timeout: float = 30.0,
                 tracer_config: Optional[Dict[str, Any]] = None):
        """Initialize TaskWorker.

        Args:
            queue: RedisTaskQueue instance to pull tasks from
            worker_id: Unique identifier for this worker
            max_concurrent_tasks: Maximum number of concurrent tasks
            poll_interval: Polling interval in seconds
            graceful_shutdown_timeout: Timeout for graceful shutdown
            tracer_config: Tracer configuration dict with "type" key
                          {"type": "langfuse", "enable_runtime_graph": False, ...}
        """
        if not isinstance(queue, DistributedTaskQueue):
            raise ValueError("TaskWorker requires a RedisTaskQueue instance")

        self.queue = queue
        self.engine = WorkflowEngine()
        self.worker_id = worker_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.poll_interval = poll_interval
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

        # Tracer configuration
        self.tracer_config = tracer_config or {}

        # Worker state
        self.is_running = False
        self.is_stopping = False
        self._worker_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        # Active tasks tracking
        self._active_tasks: Set[str] = set()
        self._active_tasks_lock = threading.Lock()

        # Metrics
        self.tasks_processed = 0
        self.tasks_succeeded = 0
        self.tasks_failed = 0
        self.tasks_timeout = 0
        self.total_execution_time = 0.0
        self.start_time = 0.0
        self._metrics_lock = threading.Lock()

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Worker {self.worker_id} received signal {signum}, initiating graceful shutdown")
            self.stop()

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Signal handlers can only be set from main thread
            logger.debug("Could not set signal handlers (not in main thread)")

    def start(self) -> None:
        """Start the worker thread."""
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} is already running")
            return

        self.is_running = True
        self.is_stopping = False
        self.start_time = time.time()

        # Initialize thread pool executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_tasks,
            thread_name_prefix=f"worker-{self.worker_id}"
        )

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"worker-{self.worker_id}-main",
            daemon=True
        )
        self._worker_thread.start()

        logger.info(f"TaskWorker {self.worker_id} started")

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the worker gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self.is_running:
            logger.warning(f"Worker {self.worker_id} is not running")
            return

        timeout = timeout or self.graceful_shutdown_timeout
        logger.info(f"Stopping worker {self.worker_id} (timeout: {timeout}s)")

        self.is_stopping = True

        # Wait for active tasks to complete
        self._wait_for_active_tasks(timeout)

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning(f"TaskWorker {self.worker_id} did not stop within timeout")

        self.is_running = False

        # Final statistics
        runtime = time.time() - self.start_time
        logger.info(
            f"Worker stopped: {self.worker_id} "
            f"(runtime: {runtime:.1f}s, tasks: {self.tasks_processed})"
        )

    def _wait_for_active_tasks(self, timeout: float) -> None:
        """Wait for active tasks to complete.

        Args:
            timeout: Maximum time to wait
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._active_tasks_lock:
                remaining_tasks = len(self._active_tasks)

            if remaining_tasks == 0:
                logger.info("All active tasks completed")
                return

            logger.debug(f"Waiting for {remaining_tasks} active tasks to complete...")
            time.sleep(1.0)

        # Timeout reached
        with self._active_tasks_lock:
            remaining_tasks = len(self._active_tasks)

        if remaining_tasks > 0:
            logger.warning(f"Shutdown timeout reached, {remaining_tasks} tasks still active")

    def _worker_loop(self) -> None:
        """Main worker loop - polls for tasks and processes them."""
        logger.info(f"Worker loop started: {self.worker_id}")

        while self.is_running and not self.is_stopping:
            try:
                # Check if we can accept more tasks
                with self._active_tasks_lock:
                    active_count = len(self._active_tasks)

                if active_count >= self.max_concurrent_tasks:
                    logger.debug(f"Max concurrent tasks reached ({active_count}), waiting...")
                    time.sleep(self.poll_interval)
                    continue

                # Try to get a task
                task_spec = self.queue.dequeue()

                if task_spec is None:
                    # No tasks available, wait before next poll
                    time.sleep(self.poll_interval)
                    continue

                # Submit task for processing
                self._submit_task(task_spec)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        logger.info(f"Worker loop finished: {self.worker_id}")

    def _submit_task(self, task_spec: TaskSpec) -> None:
        """Submit task for processing in thread pool.

        Args:
            task_spec: TaskSpec to process
        """
        if not self._executor:
            logger.error("Executor not available for task submission")
            return

        task_id = task_spec.task_id

        # Add to active tasks
        with self._active_tasks_lock:
            self._active_tasks.add(task_id)

        # Submit to thread pool
        future = self._executor.submit(self._process_task_wrapper, task_spec)

        # Add callback to handle completion
        future.add_done_callback(lambda f: self._task_completed(task_spec, f))

        logger.debug(f"Task submitted: {task_id}")

    def _create_tracer(self) -> 'Tracer':
        """Create tracer from worker configuration.

        Returns:
            Initialized tracer instance (defaults to NoopTracer)
        """
        from graflow.trace.noop import NoopTracer

        # Default to noop tracer if type not specified
        tracer_type = self.tracer_config.get("type", "noop")
        tracer_type = tracer_type.lower()

        # Extract config without "type" key
        config = {k: v for k, v in self.tracer_config.items() if k != "type"}

        try:
            if tracer_type == "noop":
                return NoopTracer(**config)
            elif tracer_type == "console":
                from graflow.trace.console import ConsoleTracer
                return ConsoleTracer(**config)
            elif tracer_type == "langfuse":
                from graflow.trace.langfuse import LangFuseTracer
                # LangFuseTracer loads API keys from .env in worker process
                return LangFuseTracer(**config)
            else:
                logger.warning(f"Unknown tracer type: {tracer_type}, using NoopTracer")
                return NoopTracer()
        except (ImportError, ValueError, TypeError) as e:
            logger.error(f"Failed to create tracer {tracer_type}: {e}")
            return NoopTracer()
        except Exception as e:
            logger.error(f"Unexpected error creating tracer {tracer_type}: {e}")
            return NoopTracer()

    def _process_task_wrapper(self, task_spec: TaskSpec) -> Dict[str, Any]:
        """Wrapper for task processing with timeout and error handling.

        Args:
            task_spec: TaskSpec to process

        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        task_id = task_spec.task_id

        try:
            # Get task from TaskSpec (already resolved from graph)
            task_func = task_spec.executable
            if task_func is None:
                raise GraflowRuntimeError(f"Task not found in graph: {task_id}")

            # Get execution context from task spec
            execution_context = task_spec.execution_context

            # Tracer initialization from worker configuration
            tracer = self._create_tracer()

            # Set tracer on ExecutionContext
            execution_context.tracer = tracer

            # Attach to parent trace for distributed tracing
            if task_spec.trace_id:
                tracer.attach_to_trace(
                    trace_id=task_spec.trace_id,
                    parent_span_id=task_spec.parent_span_id
                )

            # Create TaskWrapper from the function
            from graflow.core.task import TaskWrapper
            task_wrapper = TaskWrapper(task_id, task_func, register_to_context=False)

            # Set execution context on the task wrapper
            task_wrapper.set_execution_context(execution_context)

            # Use WorkflowEngine to execute the task
            self.engine.execute(execution_context, start_task_id=task_id)

            # Flush tracer to ensure data is sent
            tracer.shutdown()

            duration = time.time() - start_time

            result_payload = {
                "success": True,
                "duration": duration,
                "task_id": task_id,
            }
            return result_payload

        except TimeoutError: # Changed from FutureTimeoutError to TimeoutError
            duration = time.time() - start_time
            return {
                "success": False,
                "error": "Task execution timeout",
                "duration": duration,
                "task_id": task_id,
                "timeout": True
            }
        except GraflowRuntimeError as e:
            duration = time.time() - start_time
            logger.error(f"Task {task_id} failed with GraflowRuntimeError: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "task_id": task_id
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task {task_id} failed with unexpected error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "task_id": task_id,
            }

    def _task_completed(self, task_spec: TaskSpec, future: Future) -> None:
        """Handle task completion callback.

        Args:
            task_spec: Original TaskSpec
            future: Completed future
        """
        task_id = task_spec.task_id

        # Remove from active tasks
        with self._active_tasks_lock:
            self._active_tasks.discard(task_id)

        try:
            result = future.result()
            success = result.get("success", False)
            duration = result.get("duration", 0.0)
            is_timeout = result.get("timeout", False)
            error_message = result.get("error") if not success else None

            # Update metrics
            self._update_metrics(success, duration, is_timeout)

            # Notify task completion via RedisTaskQueue for barrier synchronization
            if task_spec.group_id:
                self.queue.notify_task_completion(
                    task_id, success, task_spec.group_id, error_message
                )

            if success:
                logger.info(f"Task {task_id} completed successfully")
            elif is_timeout:
                logger.warning(f"Task {task_id} timed out after {duration:.3f}s")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Task {task_id} failed: {error}")

        except Exception as e:
            logger.error(f"Error processing task completion for {task_id}: {e}")
            self._update_metrics(False, 0.0)

    def _update_metrics(self, success: bool, duration: float, is_timeout: bool = False) -> None:
        """Update worker metrics.

        Args:
            success: Whether the task succeeded
            duration: Task execution duration
            is_timeout: Whether the task timed out
        """
        with self._metrics_lock:
            self.tasks_processed += 1
            self.total_execution_time += duration

            if success:
                self.tasks_succeeded += 1
            elif is_timeout:
                self.tasks_timeout += 1
            else:
                self.tasks_failed += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics.

        Returns:
            Dictionary containing worker performance metrics
        """
        with self._metrics_lock:
            if self.tasks_processed == 0:
                avg_time = 0.0
                success_rate = 0.0
            else:
                avg_time = self.total_execution_time / self.tasks_processed
                success_rate = self.tasks_succeeded / self.tasks_processed

            with self._active_tasks_lock:
                active_count = len(self._active_tasks)

            return {
                "worker_id": self.worker_id,
                "is_running": self.is_running,
                "is_stopping": self.is_stopping,
                "active_tasks": active_count,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "tasks_processed": self.tasks_processed,
                "tasks_succeeded": self.tasks_succeeded,
                "tasks_failed": self.tasks_failed,
                "tasks_timeout": self.tasks_timeout,
                "total_execution_time": self.total_execution_time,
                "average_execution_time": avg_time,
                "success_rate": success_rate,
                "runtime": time.time() - self.start_time if self.start_time > 0 else 0.0
            }
