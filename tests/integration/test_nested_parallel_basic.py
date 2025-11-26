"""Basic integration tests for nested ParallelGroup execution validation.

Tests Phase 2 scenarios without requiring actual Worker processes:
- Nested task execution logic
- Graph hash propagation
- Dynamic task generation with nested groups
"""


from graflow.coordination.graph_store import GraphStore
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_graph_hash_generation_for_nested_groups(clean_redis):
    """
    Test: Graph hash is generated correctly for nested ParallelGroups.

    Validates that when a nested ParallelGroup is created,
    it generates a new graph hash for content-addressable storage.
    """
    # Create a simple graph
    graph = TaskGraph()

    @task
    def simple_task():
        return "result"

    graph.add_node(simple_task)

    # Initialize GraphStore
    graph_store = GraphStore(
        redis_client=clean_redis,
        key_prefix="test_nested"
    )

    # Save graph and get hash
    hash1 = graph_store.save(graph)
    assert hash1 is not None
    assert len(hash1) == 64  # SHA256 hash length

    # Same graph should return same hash (content-addressable)
    hash2 = graph_store.save(graph)
    assert hash1 == hash2

    # Different graph should return different hash
    graph2 = TaskGraph()
    @task
    def different_task():
        return "different"
    graph2.add_node(different_task)

    hash3 = graph_store.save(graph2)
    assert hash3 != hash1

def test_execution_context_graph_hash_propagation(clean_redis):
    """
    Test: ExecutionContext propagates graph_hash correctly.

    When a nested ParallelGroup is executed, the new graph hash
    should be set in the ExecutionContext.
    """
    graph = TaskGraph()
    graph_store = GraphStore(
        redis_client=clean_redis,
        key_prefix="test_propagation"
    )

    # Create context
    context = ExecutionContext(graph)

    # Simulate graph hash assignment (as done in RedisCoordinator.execute_group)
    graph_hash = graph_store.save(graph)
    context.graph_hash = graph_hash

    # Verify hash is stored
    assert hasattr(context, 'graph_hash')
    assert context.graph_hash == graph_hash

    # Load graph back using hash
    loaded_graph = graph_store.load(graph_hash)
    assert loaded_graph is not None

def test_next_task_adds_to_local_graph():
    """
    Test: next_task() adds tasks to the local graph without Redis upload.

    Worker-local task additions should modify the graph in memory
    without triggering Redis uploads (Lazy Upload principle).
    """
    graph = TaskGraph()
    context = ExecutionContext(graph)

    @task
    def initial_task():
        return "initial"

    @task
    def dynamic_task():
        return "dynamic"

    # Add initial task to graph
    graph.add_node(initial_task)
    assert "initial_task" in graph.nodes
    assert "dynamic_task" not in graph.nodes

    # Simulate next_task() behavior (adds to local graph)
    context.add_to_queue(dynamic_task)
    graph.add_node(dynamic_task)

    # Verify dynamic task is in local graph
    assert "dynamic_task" in graph.nodes
    assert len(graph.nodes) == 2

def test_nested_group_creates_separate_graph_snapshot(clean_redis):
    """
    Test: Nested ParallelGroup creates a separate graph snapshot.

    When Worker creates a nested ParallelGroup, it should save
    its current graph state as a new snapshot with a new hash.
    """
    # Create outer graph
    outer_graph = TaskGraph()

    @task
    def outer_task():
        return "outer"

    outer_graph.add_node(outer_task)

    graph_store = GraphStore(
        redis_client=clean_redis,
        key_prefix="test_snapshot"
    )

    # Save outer graph
    outer_hash = graph_store.save(outer_graph)

    # Create inner graph (simulating Worker's nested group)
    inner_graph = TaskGraph()

    @task
    def inner_task():
        return "inner"

    inner_graph.add_node(inner_task)

    # Save inner graph (separate snapshot)
    inner_hash = graph_store.save(inner_graph)

    # Verify separate hashes
    assert outer_hash != inner_hash

    # Both can be loaded independently
    loaded_outer = graph_store.load(outer_hash)
    loaded_inner = graph_store.load(inner_hash)

    assert "outer_task" in loaded_outer.nodes
    assert "inner_task" in loaded_inner.nodes

def test_sliding_ttl_for_long_running_workflows(clean_redis):
    """
    Test: Sliding TTL extends graph lifetime on access.

    Long-running workflows (e.g., nested executions) should not
    lose their graph snapshots due to TTL expiration.
    """
    import time

    graph = TaskGraph()

    @task
    def long_task():
        return "long"

    graph.add_node(long_task)

    # Create GraphStore with short TTL for testing
    graph_store = GraphStore(
        redis_client=clean_redis,
        key_prefix="test_ttl",
        ttl=2  # 2 seconds for testing
    )

    # Save graph
    graph_hash = graph_store.save(graph)

    # Wait 1 second
    time.sleep(1)

    # Load graph (should extend TTL)
    loaded = graph_store.load(graph_hash)
    assert loaded is not None

    # Wait another 1 second (total 2 seconds, but TTL was extended)
    time.sleep(1)

    # Should still be accessible (TTL was extended to 2s from last access)
    loaded_again = graph_store.load(graph_hash)
    assert loaded_again is not None

def test_lru_cache_prevents_memory_leak(clean_redis):
    """
    Test: LRU cache limits memory usage for long-running Workers.

    GraphStore's LRU cache should prevent unbounded memory growth
    even if many different graphs are loaded.
    """
    graph_store = GraphStore(
        redis_client=clean_redis,
        key_prefix="test_lru",
        cache_size=3  # Small cache for testing
    )

    # Create and save 5 different graphs
    hashes = []
    for i in range(5):
        graph = TaskGraph()

        # Create unique task for each graph
        @task(f"task_{i}")
        def unique_task():
            return f"result_{i}"

        graph.add_node(unique_task)
        hash_val = graph_store.save(graph)
        hashes.append(hash_val)

    # Load all 5 graphs (cache size is 3, so some will be evicted from cache)
    for hash_val in hashes:
        graph_store.load(hash_val)

    # Cache should only contain last 3 (LRU behavior)
    assert len(graph_store._local_cache) <= 3

    # All graphs should still be loadable from Redis
    for hash_val in hashes:
        loaded = graph_store.load(hash_val)
        assert loaded is not None
