"""Multiprocessing-based coordination backend for local parallel execution."""

import multiprocessing as mp
import threading
import time
from multiprocessing.synchronize import Barrier
from threading import BrokenBarrierError
from typing import Any, Dict, List, Optional

from graflow.coordination.coordinator import TaskCoordinator
from graflow.coordination.task_spec import TaskSpec
from graflow.exceptions import GraflowRuntimeError


class MultiprocessingCoordinator(TaskCoordinator):
    """Multiprocessing-based task coordination for local parallel execution."""

    def __init__(self, process_count: Optional[int] = None):
        """Initialize multiprocessing coordinator."""
        self.mp = mp
        self.manager = mp.Manager()
        self.barriers: Dict[str, Barrier] = {}
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process_count = process_count or mp.cpu_count()
        self.workers: List[mp.Process] = []
        self._lock = threading.Lock()
        self._barrier_lock = mp.Lock()
        self._shutdown = False

    def create_barrier(self, barrier_id: str, participant_count: int) -> str:
        """Create a barrier for parallel task synchronization."""
        with self._lock:
            try:
                barrier = self.mp.Barrier(participant_count)
                self.barriers[barrier_id] = barrier
                return barrier_id
            except Exception as e:
                raise GraflowRuntimeError(f"Failed to create barrier {barrier_id}: {e}") from e

    def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
        """Wait at barrier until all participants arrive."""
        if barrier_id not in self.barriers:
            return False

        try:
            barrier = self.barriers[barrier_id]
            barrier.wait(timeout)
            return True
        except (BrokenBarrierError, TimeoutError):
            return False
        except Exception:
            return False

    def signal_barrier(self, barrier_id: str) -> None:
        """Signal barrier completion (handled by wait_barrier in multiprocessing)."""
        # In multiprocessing, signaling is done through wait_barrier
        # This method is kept for interface compatibility
        pass

    def dispatch_task(self, task_spec: TaskSpec, group_id: str) -> None:
        """Dispatch task to multiprocessing queue."""
        task_data = {
            "task_id": task_spec.task_id,
            "func": task_spec.func,
            "args": getattr(task_spec, 'args', ()),
            "kwargs": getattr(task_spec, 'kwargs', {}),
            "group_id": group_id,
            "timestamp": time.time()
        }

        self.task_queue.put(task_data)

    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        if barrier_id in self.barriers:
            with self._lock:
                try:
                    barrier = self.barriers[barrier_id]
                    # Check if barrier is still active
                    if hasattr(barrier, 'n_waiting') and barrier.n_waiting > 0:
                        barrier.abort()
                    del self.barriers[barrier_id]
                except Exception:
                    # Barrier might already be cleaned up
                    pass

    def start_workers(self) -> None:
        """Start worker processes."""
        if self.workers:
            return  # Already started

        for i in range(self.process_count):
            worker = self.mp.Process(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def stop_workers(self) -> None:
        """Stop worker processes."""
        self._shutdown = True

        # Send stop signals to all workers
        for _ in self.workers:
            self.task_queue.put(None)  # Sentinel value to stop workers

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        self.workers.clear()

    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get result from result queue."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_queue_size(self) -> int:
        """Get approximate queue size."""
        return self.task_queue.qsize()

    def is_queue_empty(self) -> bool:
        """Check if task queue is empty."""
        return self.task_queue.empty()

    def _worker_loop(self, worker_id: int) -> None:
        """Worker process main loop."""
        while not self._shutdown:
            try:
                # Get task from queue with timeout
                task_data = self.task_queue.get(timeout=1)

                # Check for shutdown signal
                if task_data is None:
                    break

                # Execute task
                try:
                    result = self._execute_task(task_data)

                    # Put result in result queue
                    self.result_queue.put({
                        "task_id": task_data["task_id"],
                        "group_id": task_data["group_id"],
                        "result": result,
                        "worker_id": worker_id,
                        "status": "success"
                    })

                except Exception as e:
                    # Put error in result queue
                    self.result_queue.put({
                        "task_id": task_data.get("task_id", "unknown"),
                        "group_id": task_data.get("group_id", "unknown"),
                        "error": str(e),
                        "worker_id": worker_id,
                        "status": "error"
                    })

            except Exception:
                # Timeout or other exception - continue loop
                continue

    def _execute_task(self, task_data: Dict[str, Any]) -> Any:
        """Execute a single task."""
        func = task_data["func"]
        args = task_data.get("args", ())
        kwargs = task_data.get("kwargs", {})

        # Call the function
        return func(*args, **kwargs)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_workers()
        except Exception:
            pass
