# Barrier Synchronization Fix - Summary

**Date**: 2025-11-27
**Issue**: `wait_barrier()` hangs forever waiting on `pubsub.listen()` after TaskWorker finishes processing

## Root Causes

### 1. Incorrect Barrier Counting
**Problem**: The producer was incrementing the barrier counter when it shouldn't be.

According to the BSP (Bulk Synchronous Parallel) model in the design doc:
- **Producer**: Dispatches all tasks, then waits for completion (no incrementing)
- **Workers**: Execute tasks and increment barrier counter via `record_task_completion()`

**Old buggy behavior** with 3 tasks:
```
1. create_barrier(group_id, 3) → expected=3, counter=0
2. Producer calls wait_barrier() → increments to 1
3. Worker 1 completes → increments to 2
4. Worker 2 completes → increments to 3, publishes "complete"
5. Worker 3 hasn't finished yet! (barrier completed early)
```

**New correct behavior** with 3 tasks:
```
1. create_barrier(group_id, 3) → expected=3, counter=0
2. Producer calls wait_barrier() → subscribes and waits (no increment)
3. Worker 1 completes → increments to 1
4. Worker 2 completes → increments to 2
5. Worker 3 completes → increments to 3, publishes "complete"
6. Producer receives message and returns True ✓
```

### 2. Race Condition - Lost Wakeup
**Problem**: If workers complete before the producer subscribes, the publish message is lost.

**Timeline of race condition**:
```
1. Producer creates barrier
2. Producer dispatches 3 tasks
3. Workers are FAST and complete immediately
4. Workers call record_task_completion() and publish "complete"
5. Producer NOW subscribes to channel (too late!)
6. Producer waits forever in pubsub.listen() (missed the message)
```

**Fix**: Subscribe BEFORE checking if complete
```python
# Subscribe to channel first
pubsub = self.redis.pubsub()
pubsub.subscribe(barrier_info["channel"])

# THEN check if already complete
current_count_bytes = self.redis.get(barrier_info["key"])
if current_count_bytes and int(current_count_bytes) >= barrier_info["expected"]:
    return True  # Fast workers case

# If not complete, start listening (we're already subscribed)
for message in pubsub.listen():
    ...
```

## Changes Made

### 1. `graflow/coordination/redis.py`

**`wait_barrier()` method**:
- ✅ **Removed** producer's `self.redis.incr()` call
- ✅ **Subscribe immediately** before checking counter
- ✅ **Check if already complete** after subscribing (handles fast workers)
- ✅ **Updated docstring** to clarify BSP model

**Before**:
```python
def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
    # Increment participant count atomically
    current_count = self.redis.incr(barrier_info["key"])

    if current_count >= barrier_info["expected"]:
        # Last participant - notify all
        self.redis.publish(...)
        return True
    else:
        # Wait for completion
        pubsub.subscribe(...)  # TOO LATE!
```

**After**:
```python
def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
    # Subscribe FIRST (prevents race condition)
    pubsub = self.redis.pubsub()
    pubsub.subscribe(barrier_info["channel"])

    # Check if already complete (handles fast workers)
    current_count_bytes = self.redis.get(barrier_info["key"])
    if current_count_bytes and int(current_count_bytes) >= expected:
        return True

    # Wait for completion (already subscribed)
    for message in pubsub.listen():
        ...
```

### 2. Test Updates

**Updated tests to match BSP model**:
- `tests/coordination/test_coordinators.py` - Updated unit tests (mocks)
- `tests/coordination/test_redis_integration.py` - Updated integration tests (real Redis)
- Added new test: `test_barrier_race_condition_fast_workers` - Tests fast workers case

**Test pattern changes**:
```python
# OLD (incorrect): Multiple threads call wait_barrier()
def participant():
    coordinator.wait_barrier("barrier_id")  # Each increments

# NEW (correct): Producer waits, workers increment
def producer_wait():
    coordinator.wait_barrier("barrier_id")  # Only subscribes and waits

def worker():
    record_task_completion(...)  # Only workers increment
```

## Verification

### Unit Tests (no Redis needed)
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/coordination/test_coordinators.py::TestRedisCoordinator -v
# Result: 12 passed
```

### Integration Tests (requires Redis)
```bash
# Start Redis
docker run -p 6379:6379 redis:7.2

# Run integration tests
PYTHONPATH=. .venv/bin/python -m pytest tests/coordination/test_redis_integration.py -v -m integration
```

## Summary

The fix implements the correct BSP (Bulk Synchronous Parallel) model:

1. **Producer**: Dispatches all tasks → subscribes → waits (no counting)
2. **Workers**: Execute tasks → increment barrier → last worker publishes
3. **Race condition handled**: Subscribe before checking completion status

This ensures the barrier waits for **all workers** to complete, and handles the race condition where workers finish before the producer subscribes.
