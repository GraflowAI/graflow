# Redis Client Serialization Fix

**Status:** Design Phase
**Author:** System Design
**Date:** 2025-12-06
**Related Issues:** Serialization errors with nested ParallelGroups using Redis backend

## Problem Statement

When using Redis-backed parallel execution with nested `ParallelGroup` instances, task graph serialization fails with:

```
GraflowRuntimeError: An error occurred: cannot pickle '_thread.lock' object
```

The lock detection tool reveals:

```
Found 5 thread lock(s) in TaskGraph:
  Lock at: root._graph._node['L1_group']['task']._execution_config['backend_config']['redis_client'].single_connection_lock
  Lock at: root._graph._node['L1_group']['task']._execution_config['backend_config']['redis_client']._event_dispatcher._lock
  Lock at: root._graph._node['L1_group']['task']._execution_config['backend_config']['redis_client'].connection_pool._fork_lock
  Lock at: root._graph._node['L1_group']['task']._execution_config['backend_config']['redis_client'].connection_pool._lock
  Lock at: root._graph._node['L1_group']['task']._execution_config['backend_config']['redis_client'].connection_pool._event_dispatcher._lock
```

## Root Cause Analysis

### Current Flow

1. User creates ParallelGroup with Redis backend:
   ```python
   parallel_group = (task_a | task_b).with_execution(
       CoordinationBackend.REDIS,
       backend_config={"redis_client": redis_client, "key_prefix": "test"}
   )
   ```

2. `ParallelGroup.with_execution()` stores `redis_client` in `_execution_config`:
   ```python
   # task.py:399-448
   def with_execution(self, backend_config: Optional[dict] = None, ...):
       if backend_config is not None:
           self._execution_config["backend_config"].update(backend_config)  # ❌ Stores redis_client
   ```

3. ParallelGroup is added to TaskGraph:
   ```python
   # Graph stores the entire ParallelGroup object with redis_client inside
   graph.add_node(parallel_group, task_id)
   ```

4. When TaskGraph is serialized, networkx pickles all node data including `redis_client` → **FAILS**

### Why Existing __getstate__() Doesn't Help

While `ParallelGroup.__getstate__()` (task.py:575-611) correctly extracts redis connection params:

```python
def __getstate__(self) -> dict:
    # ... extracts host, port, db from redis_client ...
    backend_config.pop("redis_client")
    backend_config.update(redis_config)
```

**This only works when `ParallelGroup` itself is pickled directly.** When TaskGraph is pickled, networkx's internal serialization bypasses individual `__getstate__()` methods on nested objects within the graph structure.

### Architecture Issue

```
TaskGraph (pickled)
  └── networkx._graph._node[task_id]['task']  ← Stored as-is
        └── ParallelGroup
              └── _execution_config['backend_config']['redis_client']  ❌ Thread locks!
```

The `redis_client` should **never be stored** in any serializable object's internal state.

## Design Options

### Option 1: Fix at TaskGraph Level ❌

**Approach:** Modify `TaskGraph.__getstate__()` to recursively clean all task objects.

**Problems:**
- Complex: Must traverse networkx graph structure and manually call `__getstate__()` on all tasks
- Error-prone: Easy to miss nested objects
- Tight coupling: TaskGraph needs to know about internal structure of all task types
- Not future-proof: New task types must be explicitly handled

**Verdict:** Too complex and brittle.

---

### Option 2: Prevent redis_client Storage ✅ **RECOMMENDED**

**Approach:** Normalize `backend_config` immediately when set, converting `redis_client` → connection params.

**Key Principle:** Never store unpicklable objects in serializable state.

**Implementation:**

1. **Create `graflow/utils/redis_utils.py`:**
   ```python
   def extract_redis_config(redis_client: Redis) -> Dict[str, Any]:
       """Extract connection params from Redis client."""
       return {
           'host': redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
           'port': redis_client.connection_pool.connection_kwargs.get('port', 6379),
           'db': redis_client.connection_pool.connection_kwargs.get('db', 0),
           'decode_responses': getattr(redis_client, 'decode_responses', False),
       }

   def create_redis_client(config: Dict[str, Any]) -> Redis:
       """Create Redis client from connection params."""
       return Redis(
           host=config.get('host', 'localhost'),
           port=config.get('port', 6379),
           db=config.get('db', 0),
           decode_responses=config.get('decode_responses', False),
       )

   def normalize_redis_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
       """Convert redis_client → connection params.

       Raises:
           ValueError: If redis_client extraction fails (fail-fast approach)
       """
       if config is None or 'redis_client' not in config:
           return config or {}

       config = config.copy()
       redis_client = config.pop('redis_client')

       try:
           redis_config = extract_redis_config(redis_client)
           config.update(redis_config)
       except Exception as e:
           # Fail fast: Don't silently use defaults
           raise ValueError(
               f"Failed to extract Redis connection parameters from redis_client: {e}"
           ) from e

       return config
   ```

2. **Normalize at entry points:**

   **`ParallelGroup.with_execution()`:**
   ```python
   def with_execution(self, backend_config: Optional[dict] = None, ...):
       if backend_config is not None:
           from graflow.utils.redis_utils import normalize_redis_config
           backend_config = normalize_redis_config(backend_config)  # ✅ Remove redis_client
           self._execution_config["backend_config"].update(backend_config)
   ```

   **`ExecutionContext.__init__()`:**
   ```python
   def __init__(self, config: Optional[Dict[str, Any]] = None, ...):
       from graflow.utils.redis_utils import normalize_redis_config
       self._original_config = normalize_redis_config(config)  # ✅ Remove redis_client
   ```

3. **Create redis_client on-demand:**

   **`GroupExecutor._create_coordinator()`:**
   ```python
   if backend == CoordinationBackend.REDIS:
       from graflow.utils.redis_utils import create_redis_client

       # Create client from connection params (never stored)
       redis_client = create_redis_client(config)

       redis_kwargs = {
           "redis_client": redis_client,
           "key_prefix": config.get("key_prefix", "graflow")
       }
       task_queue = DistributedTaskQueue(**redis_kwargs)
   ```

4. **Simplify/remove `__getstate__()` redis handling:**
   - No longer needed in `ParallelGroup.__getstate__()`
   - No longer needed in `ExecutionContext.__getstate__()`
   - Already removed from `TaskGraph.__getstate__()`

**Benefits:**
- ✅ **Prevents problem at source** - redis_client never enters serializable objects
- ✅ **Reusable utilities** - `normalize_redis_config()` can be used everywhere
- ✅ **Less error-prone** - No complex recursion or graph traversal
- ✅ **Easy to test** - Simple utility functions
- ✅ **Backwards compatible** - Existing code that passes `redis_client` still works
- ✅ **Simpler codebase** - Remove complex `__getstate__()` workarounds
- ✅ **Fail-fast design** - Errors in config extraction are caught immediately, not hidden

**Error Handling Philosophy:**

**Fail Fast, Not Silent Defaults**

If extracting connection params from `redis_client` fails, we raise `ValueError` immediately rather than silently falling back to defaults (localhost:6379).

**Rationale:**
- Silent defaults could connect to wrong Redis instance → data corruption
- Bugs are caught immediately at config time, not later during execution
- Clear error messages help debugging
- Explicit is better than implicit

**Example:**
```python
# If redis_client is corrupted/invalid
config = {'redis_client': broken_client}
normalize_redis_config(config)  # Raises ValueError immediately ✅

# Instead of silently using localhost:6379 which might be wrong ❌
```

**Drawbacks:**
- Requires changes in multiple files
- Need to ensure all entry points are covered

---

### Option 3: Lazy Client Creation Everywhere ❌

**Approach:** Never accept `redis_client` in configs - always create from connection params.

**Changes:**
- Update API: `backend_config={"host": "...", "port": ...}` only
- Remove all `redis_client` handling
- Create client lazily everywhere it's needed

**Problems:**
- ❌ **Breaking API change** - Existing code will break
- ❌ **Most invasive** - Requires updating all examples, tests, docs
- ❌ **Less flexible** - Users can't provide pre-configured clients

**Verdict:** Too disruptive for marginal benefit.

---

## Recommended Solution: Option 2

**Implementation Strategy:** Normalize at entry, create on-demand

## Implementation Plan

### Phase 1: Create Utilities ✅

**File:** `graflow/utils/redis_utils.py`

**Functions:**
- `extract_redis_config(redis_client: Redis) -> Dict[str, Any]`
- `create_redis_client(config: Dict[str, Any]) -> Redis`
- `normalize_redis_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]`

**Tests:** `tests/unit/test_redis_utils.py`
- Test extraction from real Redis client
- Test client creation from params
- Test normalization with/without redis_client
- Test error handling (invalid client, missing params)

### Phase 2: Update Entry Points

#### 2.1 ParallelGroup (`graflow/core/task.py`)

**Location:** `ParallelGroup.with_execution()` (line 399-448)

**Change:**
```python
def with_execution(
    self,
    backend: Optional[CoordinationBackend] = None,
    backend_config: Optional[dict] = None,
    policy: Union[str, GroupExecutionPolicy] = "strict",
) -> ParallelGroup:
    if backend is not None:
        self._execution_config["backend"] = backend

    if backend_config is not None:
        # ✅ NEW: Normalize config to remove redis_client
        from graflow.utils.redis_utils import normalize_redis_config
        backend_config = normalize_redis_config(backend_config)

        self._execution_config["backend_config"].update(backend_config)

    # ... rest unchanged
```

#### 2.2 ExecutionContext (`graflow/core/context.py`)

**Location:** `ExecutionContext.__init__()` (around line 500-600)

**Change:**
```python
def __init__(
    self,
    graph: TaskGraph,
    start_node: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    # ... other params
):
    # ✅ NEW: Normalize config to remove redis_client
    from graflow.utils.redis_utils import normalize_redis_config
    self._original_config = normalize_redis_config(config or {})

    # ... rest unchanged (now uses self._original_config without redis_client)
```

### Phase 3: Update Consumers

#### 3.1 GroupExecutor (`graflow/coordination/executor.py`)

**Location:** `GroupExecutor._create_coordinator()` (line 48-75)

**Change:**
```python
@staticmethod
def _create_coordinator(
    backend: CoordinationBackend,
    config: Dict[str, Any]
) -> GroupCoordinator:
    if backend == CoordinationBackend.REDIS:
        from graflow.utils.redis_utils import create_redis_client

        # ✅ NEW: Create client from connection params
        redis_client = create_redis_client(config)

        redis_kwargs = {
            "redis_client": redis_client,
            "key_prefix": config.get("key_prefix", "graflow")
        }

        try:
            task_queue = DistributedTaskQueue(**redis_kwargs)
        except ImportError as e:
            raise ImportError("Redis backend requires 'redis' package") from e

        return RedisCoordinator(task_queue)

    # ... rest unchanged
```

#### 3.2 Other Redis Consumers

Search for `redis_client` usage and update:
- `graflow/channels/redis.py` - RedisChannel creation
- `graflow/queue/redis.py` - RedisTaskQueue creation
- `graflow/hitl/backend/redis.py` - RedisFeedbackBackend creation

**Pattern:**
```python
# OLD: Accept redis_client directly
def __init__(self, redis_client: Optional[Redis] = None, host="localhost", ...):
    if redis_client:
        self.redis = redis_client
    else:
        self.redis = Redis(host=host, ...)

# NEW: Accept config, create client
def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
    from graflow.utils.redis_utils import create_redis_client
    config = {**config, **kwargs} if config else kwargs
    self.redis = create_redis_client(config)
```

**Note:** This is optional - may keep existing APIs for now and only change internal storage.

### Phase 4: Cleanup __getstate__() Methods

**Files to update:**

1. **`graflow/core/task.py`** - `ParallelGroup.__getstate__()`
   - Remove redis_client extraction logic (lines 589-604)
   - Simplify to just copy state (redis_client already removed in `with_execution()`)

2. **`graflow/core/context.py`** - `ExecutionContext.__getstate__()`
   - Remove redis_client extraction logic (lines 1255-1275)
   - Simplify (redis_client already removed in `__init__()`)

**Simplified version:**
```python
def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    # Remove un-serializable runtime objects only
    state.pop('task_queue', None)
    state.pop('channel', None)
    state.pop('feedback_manager', None)
    state['_llm_agents'] = {}  # Agent instances not serialized
    # ✅ No redis_client handling needed - already normalized
    return state
```

### Phase 5: Update Tests

**New test file:** `tests/unit/test_redis_utils.py`
```python
import pytest
from graflow.utils.redis_utils import (
    extract_redis_config,
    create_redis_client,
    normalize_redis_config
)

def test_extract_redis_config():
    """Test extracting connection params from Redis client."""
    import redis
    client = redis.Redis(host='testhost', port=1234, db=5, decode_responses=True)
    config = extract_redis_config(client)

    assert config['host'] == 'testhost'
    assert config['port'] == 1234
    assert config['db'] == 5
    assert config['decode_responses'] is True

def test_create_redis_client():
    """Test creating Redis client from config."""
    config = {'host': 'localhost', 'port': 6379, 'db': 0}
    client = create_redis_client(config)

    assert client.connection_pool.connection_kwargs['host'] == 'localhost'
    assert client.connection_pool.connection_kwargs['port'] == 6379

def test_normalize_redis_config():
    """Test normalizing config with redis_client."""
    import redis
    client = redis.Redis(host='localhost', port=6379)
    config = {'redis_client': client, 'key_prefix': 'test'}

    normalized = normalize_redis_config(config)

    # redis_client should be removed
    assert 'redis_client' not in normalized
    # Connection params should be extracted
    assert normalized['host'] == 'localhost'
    assert normalized['port'] == 6379
    # Other keys should be preserved
    assert normalized['key_prefix'] == 'test'

def test_normalize_redis_config_without_client():
    """Test normalizing config without redis_client."""
    config = {'host': 'localhost', 'port': 6379}
    normalized = normalize_redis_config(config)

    # Should return unchanged
    assert normalized == config

def test_normalize_redis_config_with_none():
    """Test normalizing None config."""
    normalized = normalize_redis_config(None)
    assert normalized == {}

def test_normalize_redis_config_extraction_failure():
    """Test fail-fast behavior when extraction fails."""
    # Mock an invalid redis_client object
    class InvalidClient:
        pass

    config = {'redis_client': InvalidClient()}

    # Should raise ValueError, not silently use defaults
    with pytest.raises(ValueError, match="Failed to extract Redis connection parameters"):
        normalize_redis_config(config)

def test_extract_redis_config_invalid_client():
    """Test extraction with invalid client object."""
    class FakeClient:
        pass

    # Should raise AttributeError (no connection_pool)
    with pytest.raises(AttributeError):
        extract_redis_config(FakeClient())
```

**Update integration tests:** `tests/integration/test_nested_parallel_execution.py`
- Add test to verify serialization works after fix
- Pickle/unpickle TaskGraph with nested ParallelGroups

```python
def test_nested_parallel_serialization(clean_redis):
    """Test that nested parallel groups can be serialized."""
    import pickle
    from graflow.core.task import ParallelGroup, Task
    from graflow.core.graph import TaskGraph

    # Create nested parallel groups with Redis backend
    inner_group = (task1 | task2).with_execution(
        CoordinationBackend.REDIS,
        backend_config={'redis_client': clean_redis}
    )
    outer_group = (inner_group | task3).with_execution(
        CoordinationBackend.REDIS,
        backend_config={'redis_client': clean_redis}
    )

    # Add to graph
    graph = TaskGraph()
    graph.add_node(outer_group, "outer")

    # Should serialize without errors
    pickled = pickle.dumps(graph)
    restored = pickle.loads(pickled)

    # Verify structure preserved
    assert "outer" in restored.nodes
```

## Migration Guide

### For Users

**No breaking changes!** Existing code continues to work:

```python
# ✅ BEFORE: Still works
parallel = (task_a | task_b).with_execution(
    CoordinationBackend.REDIS,
    backend_config={"redis_client": redis_client}
)

# ✅ AFTER: Also works (recommended)
parallel = (task_a | task_b).with_execution(
    CoordinationBackend.REDIS,
    backend_config={"host": "localhost", "port": 6379, "db": 0}
)
```

The difference: `redis_client` is automatically converted to connection params internally.

**Important:** If `redis_client` extraction fails, you'll now get a clear error instead of silent defaults:

```python
# Invalid/corrupted redis_client
config = {"redis_client": broken_client}
parallel.with_execution(backend_config=config)
# Raises: ValueError: Failed to extract Redis connection parameters from redis_client

# This is better than silently connecting to localhost:6379!
```

### For Developers

**When adding new Redis-backed features:**

1. **Accept connection params, not client:**
   ```python
   # ✅ GOOD
   def my_feature(config: Dict[str, Any]):
       from graflow.utils.redis_utils import create_redis_client
       redis_client = create_redis_client(config)

   # ❌ BAD: Don't store redis_client in serializable objects
   class MyTask(Executable):
       def __init__(self, redis_client):
           self.redis_client = redis_client  # ❌ Cannot pickle!
   ```

2. **Use normalization at config entry points:**
   ```python
   def with_config(self, config: dict):
       from graflow.utils.redis_utils import normalize_redis_config
       self.config = normalize_redis_config(config)
   ```

## Testing Strategy

### Unit Tests
- ✅ `test_redis_utils.py` - Test all utility functions
- ✅ Test error handling (missing Redis package, invalid config)

### Integration Tests
- ✅ Test nested ParallelGroup serialization
- ✅ Test ExecutionContext serialization with Redis backend
- ✅ Test workflow execution after deserialization

### Manual Testing
1. Run `tests/integration/test_nested_parallel_execution.py`
2. Verify no lock detection warnings
3. Verify workflows execute correctly

## Success Criteria

- [ ] No `cannot pickle '_thread.lock'` errors
- [ ] Lock detection tool shows 0 locks in serialized objects
- [ ] All integration tests pass
- [ ] Backwards compatibility maintained (existing tests pass)
- [ ] Code coverage ≥ 90% for new utilities

## Rollout Plan

1. **Phase 1:** Implement utilities + tests (1 day)
2. **Phase 2:** Update ParallelGroup + ExecutionContext (1 day)
3. **Phase 3:** Update GroupExecutor + consumers (1 day)
4. **Phase 4:** Cleanup __getstate__() methods (0.5 day)
5. **Phase 5:** Integration testing + documentation (0.5 day)

**Total:** ~3-4 days

## Future Enhancements

1. **Deprecation warning:** Add warning when `redis_client` is passed (guide users to connection params)
2. **Connection pooling:** Reuse Redis clients across tasks (requires careful lifecycle management)
3. **Async Redis:** Support async Redis clients for better performance

## References

- Issue: Task serialization fails with nested ParallelGroups
- Related: `graflow/debug/find_locks.py` - Lock detection utility
- Related: `graflow/core/serialization.py` - Serialization helpers
