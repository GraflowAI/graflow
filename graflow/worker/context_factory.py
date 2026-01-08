"""Factory for creating ExecutionContext from serialized records."""

from typing import Tuple

from graflow.coordination.graph_store import GraphStore
from graflow.coordination.records import SerializedTaskRecord
from graflow.core.context import ExecutionContext
from graflow.core.task import Executable


class ExecutionContextFactory:
    """Reconstruct ExecutionContext and Task on the worker side."""

    @staticmethod
    def create_from_record(
        record: SerializedTaskRecord, graph_store: GraphStore
    ) -> Tuple[ExecutionContext, Executable]:
        """Reconstruct ExecutionContext and Executable from SerializedTaskRecord.

        Args:
            record: TaskRecord retrieved from Redis
            graph_store: GraphStore instance

        Returns:
            (ExecutionContext, Executable)

        Raises:
            ValueError: If graph or task not found
        """
        # 1. Get Graph from graph_hash
        graph = graph_store.load(record.graph_hash)
        # graph_store.load() raises ValueError if not found

        # 2. Get task from Graph
        try:
            task = graph.get_node(record.task_id)
        except KeyError:
            task = None

        if task is None:
            raise ValueError(f"Task {record.task_id} not found in graph {record.graph_hash}")

        # 3. Create ExecutionContext
        context = ExecutionContext(
            graph=graph,
            start_node=None,
            session_id=record.session_id,
            trace_id=record.trace_id,
            channel_backend="redis",
            config={
                "redis_client": graph_store.redis,
                "key_prefix": graph_store.key_prefix,
            },
        )

        # Set graph_hash in context so it can be propagated to children
        context.graph_hash = record.graph_hash

        return context, task
