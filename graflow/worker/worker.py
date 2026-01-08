"""Task worker implementation for processing tasks from queues."""

import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from graflow.core.engine import WorkflowEngine
from graflow.exceptions import GraflowRuntimeError
from graflow.queue.base import TaskSpec
from graflow.queue.distributed import DistributedTaskQueue

if TYPE_CHECKING:
    from graflow.trace.base import Tracer


class TaskWorker:
    """Worker that processes tasks from a queue using WorkflowEngine."""

    def __init__(
        self,
        queue: DistributedTaskQueue,
        worker_id: str,
        max_concurrent_tasks: int = 4,
        poll_interval: float = 0.1,
        graceful_shutdown_timeout: float = 30.0,
        tracer_config: Optional[Dict[str, Any]] = None,
    ):
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
        self._logger.info("Initializing TaskWorker: worker_id=%s", worker_id)
        self._logger.debug(
            "Worker config: max_concurrent=%d, poll_interval=%.2fs, shutdown_timeout=%.1fs",
            max_concurrent_tasks,
            poll_interval,
            graceful_shutdown_timeout,
        )

        if not isinstance(queue, DistributedTaskQueue):
            self._logger.error("Invalid queue type: %s (expected DistributedTaskQueue)", type(queue).__name__)
            raise ValueError("TaskWorker requires a RedisTaskQueue instance")

        self.queue = queue
        self.engine = WorkflowEngine()
        self.worker_id = worker_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.poll_interval = poll_interval
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

        # Tracer configuration
        self.tracer_config = tracer_config or {}
        if self.tracer_config:
            self._logger.debug("Tracer config: %s", {k: v for k, v in self.tracer_config.items() if k != "api_key"})

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
        self._logger.info("TaskWorker '%s' initialized successfully", worker_id)

    @property
    def _logger(self):
        """Get logger instance (lazy initialization to avoid module-level import)."""
        if not hasattr(self, "_logger_instance"):
            import logging

            self._logger_instance = logging.getLogger(__name__)
        return self._logger_instance

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            self._logger.info(f"Worker {self.worker_id} received signal {signum}, initiating graceful shutdown")
            self.stop()

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Signal handlers can only be set from main thread
            self._logger.debug("Could not set signal handlers (not in main thread)")

    def start(self) -> None:
        """Start the worker thread."""
        if self.is_running:
            self._logger.warning(f"Worker {self.worker_id} is already running")
            return

        self.is_running = True
        self.is_stopping = False
        self.start_time = time.time()

        # Initialize thread pool executor
        self._logger.debug(f"Initializing ThreadPoolExecutor with {self.max_concurrent_tasks} workers")
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_tasks, thread_name_prefix=f"worker-{self.worker_id}"
        )

        # Start worker thread
        self._logger.debug(f"Starting worker thread for {self.worker_id}")
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name=f"worker-{self.worker_id}-main", daemon=True
        )
        self._worker_thread.start()

        self._logger.info(
            f"TaskWorker {self.worker_id} started (max_concurrent={self.max_concurrent_tasks}, "
            f"poll_interval={self.poll_interval}s, shutdown_timeout={self.graceful_shutdown_timeout}s)"
        )

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the worker gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self.is_running:
            self._logger.warning(f"Worker {self.worker_id} is not running")
            return

        timeout = timeout or self.graceful_shutdown_timeout
        self._logger.info(f"Stopping worker {self.worker_id} (timeout: {timeout}s)")

        self.is_stopping = True

        # Wait for active tasks to complete
        self._logger.info(f"Waiting for active tasks to complete (timeout: {timeout}s)")
        self._wait_for_active_tasks(timeout)

        # Shutdown executor
        if self._executor:
            self._logger.debug("Shutting down ThreadPoolExecutor")
            self._executor.shutdown(wait=True)
            self._executor = None
            self._logger.debug("ThreadPoolExecutor shutdown complete")

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._logger.debug("Waiting for worker thread to finish")
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                self._logger.warning(f"TaskWorker {self.worker_id} did not stop within timeout")
            else:
                self._logger.debug("Worker thread finished")

        self.is_running = False

        # Final statistics
        runtime = time.time() - self.start_time
        with self._metrics_lock:
            self._logger.info(
                f"Worker stopped: {self.worker_id} "
                f"(runtime: {runtime:.1f}s, processed: {self.tasks_processed}, "
                f"succeeded: {self.tasks_succeeded}, failed: {self.tasks_failed}, "
                f"timeout: {self.tasks_timeout})"
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
                self._logger.info("All active tasks completed")
                return

            self._logger.debug(f"Waiting for {remaining_tasks} active tasks to complete...")
            time.sleep(1.0)

        # Timeout reached
        with self._active_tasks_lock:
            remaining_tasks = len(self._active_tasks)

        if remaining_tasks > 0:
            self._logger.warning(f"Shutdown timeout reached, {remaining_tasks} tasks still active")

    def _worker_loop(self) -> None:
        """Main worker loop - polls for tasks and processes them."""
        self._logger.info(f"Worker loop started: {self.worker_id}")

        while self.is_running and not self.is_stopping:
            try:
                # Check if we can accept more tasks
                with self._active_tasks_lock:
                    active_count = len(self._active_tasks)
                    active_tasks_list = list(self._active_tasks)

                if active_count >= self.max_concurrent_tasks:
                    self._logger.debug(
                        f"Max concurrent tasks reached ({active_count}/{self.max_concurrent_tasks}), "
                        f"waiting... Active: {active_tasks_list[:3]}{'...' if len(active_tasks_list) > 3 else ''}"
                    )
                    time.sleep(self.poll_interval)
                    continue

                self._logger.debug(f"Polling queue for tasks (active: {active_count}/{self.max_concurrent_tasks})")

                # Try to get a task
                task_spec = self.queue.dequeue()

                if task_spec is None:
                    # No tasks available, wait before next poll
                    self._logger.debug(f"No tasks available in queue, polling again in {self.poll_interval}s")
                    time.sleep(self.poll_interval)
                    continue

                self._logger.info(f"Dequeued task: {task_spec.task_id} (group: {task_spec.group_id or 'None'})")

                # Submit task for processing
                self._submit_task(task_spec)

            except Exception as e:
                self._logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        self._logger.info(f"Worker loop finished: {self.worker_id}")

    def _submit_task(self, task_spec: TaskSpec) -> None:
        """Submit task for processing in thread pool.

        Args:
            task_spec: TaskSpec to process
        """
        if not self._executor:
            self._logger.error("Executor not available for task submission")
            return

        task_id = task_spec.task_id

        # Add to active tasks
        with self._active_tasks_lock:
            self._active_tasks.add(task_id)
            active_count = len(self._active_tasks)

        self._logger.debug(
            f"Submitting task {task_id} to thread pool (active: {active_count}/{self.max_concurrent_tasks})"
        )

        # Submit to thread pool
        future = self._executor.submit(self._process_task_wrapper, task_spec)

        # Add callback to handle completion
        future.add_done_callback(lambda f: self._task_completed(task_spec, f))

        self._logger.debug(f"Task {task_id} submitted successfully")

    def _create_tracer(self) -> "Tracer":
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

        self._logger.debug(f"Creating tracer: type={tracer_type}, config_keys={list(config.keys())}")

        try:
            if tracer_type == "noop":
                self._logger.debug("Initialized NoopTracer")
                return NoopTracer(**config)
            elif tracer_type == "console":
                from graflow.trace.console import ConsoleTracer

                self._logger.debug("Initialized ConsoleTracer")
                return ConsoleTracer(**config)
            elif tracer_type == "langfuse":
                from graflow.trace.langfuse import LangFuseTracer

                # LangFuseTracer loads API keys from .env in worker process
                self._logger.debug("Initialized LangFuseTracer")
                return LangFuseTracer(**config)
            else:
                self._logger.warning(f"Unknown tracer type: {tracer_type}, using NoopTracer")
                return NoopTracer()
        except (ImportError, ValueError, TypeError) as e:
            self._logger.error(f"Failed to create tracer {tracer_type}: {e}")
            return NoopTracer()
        except Exception as e:
            self._logger.error(f"Unexpected error creating tracer {tracer_type}: {e}")
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

        self._logger.info(f"Starting task execution: {task_id}")
        self._logger.debug(
            f"Task spec - group_id: {task_spec.group_id}, trace_id: {task_spec.trace_id}, "
            f"parent_span_id: {task_spec.parent_span_id}"
        )

        try:
            # Get task from TaskSpec (already resolved from graph)
            task_func = task_spec.executable
            if task_func is None:
                raise GraflowRuntimeError(f"Task not found in graph: {task_id}")

            # Get execution context from task spec
            execution_context = task_spec.execution_context

            # Tracer initialization from worker configuration
            self._logger.debug(f"Initializing tracer for task {task_id}")
            tracer = self._create_tracer()

            # Set tracer on ExecutionContext
            execution_context.tracer = tracer

            # Attach to parent trace for distributed tracing
            if task_spec.trace_id:
                self._logger.debug(
                    f"Attaching to parent trace: trace_id={task_spec.trace_id}, "
                    f"parent_span_id={task_spec.parent_span_id}"
                )
                tracer.attach_to_trace(trace_id=task_spec.trace_id, parent_span_id=task_spec.parent_span_id)

            # Create TaskWrapper from the function
            from graflow.core.task import TaskWrapper

            self._logger.debug(f"Creating TaskWrapper for task {task_id}")
            task_wrapper = TaskWrapper(task_id, task_func, register_to_context=False)

            # Set execution context on the task wrapper
            task_wrapper.set_execution_context(execution_context)

            # Use WorkflowEngine to execute the task
            self._logger.debug(f"Executing task {task_id} via WorkflowEngine")
            self.engine.execute(execution_context, start_task_id=task_id)

            # Flush tracer to ensure data is sent
            self._logger.debug(f"Shutting down tracer for task {task_id}")
            tracer.shutdown()

            duration = time.time() - start_time

            self._logger.info(f"Task {task_id} execution completed successfully in {duration:.3f}s")

            result_payload = {
                "success": True,
                "duration": duration,
                "task_id": task_id,
            }
            return result_payload

        except TimeoutError:  # Changed from FutureTimeoutError to TimeoutError
            duration = time.time() - start_time
            self._logger.warning(f"Task {task_id} execution timed out after {duration:.3f}s")
            return {
                "success": False,
                "error": "Task execution timeout",
                "duration": duration,
                "task_id": task_id,
                "timeout": True,
            }
        except GraflowRuntimeError as e:
            duration = time.time() - start_time
            self._logger.error(
                f"Task {task_id} failed with GraflowRuntimeError after {duration:.3f}s: {e}", exc_info=True
            )
            return {"success": False, "error": str(e), "duration": duration, "task_id": task_id}
        except Exception as e:
            duration = time.time() - start_time
            self._logger.error(f"Task {task_id} failed with unexpected error after {duration:.3f}s: {e}", exc_info=True)
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
            remaining_active = len(self._active_tasks)

        self._logger.debug(f"Task {task_id} completed callback triggered (active tasks: {remaining_active})")

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
                self._logger.debug(
                    f"Notifying task completion for group {task_spec.group_id}: task={task_id}, success={success}"
                )
                self.queue.notify_task_completion(task_id, success, task_spec.group_id, error_message)

            if success:
                self._logger.info(f"Task {task_id} completed successfully in {duration:.3f}s")
            elif is_timeout:
                self._logger.warning(f"Task {task_id} timed out after {duration:.3f}s")
            else:
                error = result.get("error", "Unknown error")
                self._logger.error(f"Task {task_id} failed after {duration:.3f}s: {error}")

        except Exception as e:
            self._logger.error(f"Error processing task completion for {task_id}: {e}", exc_info=True)
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
                self._logger.debug(
                    f"Metrics updated: task succeeded (total: {self.tasks_processed}, "
                    f"succeeded: {self.tasks_succeeded}, duration: {duration:.3f}s)"
                )
            elif is_timeout:
                self.tasks_timeout += 1
                self._logger.debug(
                    f"Metrics updated: task timed out (total: {self.tasks_processed}, "
                    f"timeout: {self.tasks_timeout}, duration: {duration:.3f}s)"
                )
            else:
                self.tasks_failed += 1
                self._logger.debug(
                    f"Metrics updated: task failed (total: {self.tasks_processed}, "
                    f"failed: {self.tasks_failed}, duration: {duration:.3f}s)"
                )

            # Log periodic stats every 10 tasks
            if self.tasks_processed % 10 == 0:
                avg_time = self.total_execution_time / self.tasks_processed
                success_rate = self.tasks_succeeded / self.tasks_processed
                self._logger.info(
                    f"Worker {self.worker_id} stats: processed={self.tasks_processed}, "
                    f"succeeded={self.tasks_succeeded}, failed={self.tasks_failed}, "
                    f"timeout={self.tasks_timeout}, avg_time={avg_time:.3f}s, "
                    f"success_rate={success_rate:.2%}"
                )

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
                "runtime": time.time() - self.start_time if self.start_time > 0 else 0.0,
            }
