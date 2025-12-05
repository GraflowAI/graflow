# Redis Pub/Sub + Polling Hybrid - Design Clarification

**Document**: Pub/Sub with Polling Fallback Strategy
**Date**: 2025-01-28
**Context**: Response to subscriber liveness concern

---

## Problem Statement

**Redis Pub/Sub Limitation**: Messages are NOT persisted
- If a message is published when no subscriber is listening, it is LOST
- Pub/Sub is "fire-and-forget"

**Workflow Scenario**:
```
1. Worker polling for feedback (timeout: 30s)
2. Timeout → Checkpoint created → Worker exits
3. (5 minutes later) Human provides feedback via API
4. Pub/Sub message published → NO ACTIVE SUBSCRIBER → Message lost ❌
5. Worker 2 resumes from checkpoint
6. How does Worker 2 get the feedback? 🤔
```

**Your Question**: Does "pubsub (synchronous) + polling (asynchronous)" solve this?

**Answer**: YES, exactly! ✅

---

## Solution: Pub/Sub as Optimization, Storage as Source of Truth

### Key Principle

```
┌─────────────────────────────────────────────┐
│  Redis Storage (Hash) = SOURCE OF TRUTH    │
│  ├─ feedback:request:xxx                    │
│  └─ feedback:response:xxx  ◄────────────────┼─── Always check here
└─────────────────────────────────────────────┘
           ▲
           │
           │ (Optional notification for latency optimization)
           │
┌─────────────────────────────────────────────┐
│  Redis Pub/Sub = NOTIFICATION ONLY          │
│  └─ feedback:xxx channel                    │
└─────────────────────────────────────────────┘
```

### Two Scenarios

#### Scenario A: Subscriber Active (Synchronous)

```
Timeline:
  0s: Worker starts polling (subscriber active)
  5s: Human provides feedback
  5s: Pub/Sub message published → Subscriber receives immediately
  5s: Worker fetches response from storage
  5s: Worker continues (total latency: < 100ms)
```

**Result**: Low latency ✅ (Pub/Sub optimization works)

#### Scenario B: Subscriber Inactive (Asynchronous)

```
Timeline:
  0s: Worker starts polling
 30s: Timeout → Checkpoint → Worker exits (subscriber dead)
  5min: Human provides feedback
  5min: Pub/Sub message published → NO SUBSCRIBER → Lost ❌
  10min: Worker 2 resumes from checkpoint
  10min: Worker 2 checks storage → Response found ✅
  10min: Worker 2 continues
```

**Result**: Reliable resume ✅ (Polling fallback works)

---

## Implementation Details

### Hybrid Polling Loop

```python
def _poll_for_response(self, feedback_id: str, timeout: float):
    """Polling with Pub/Sub optimization (hybrid approach)."""

    # Storage is always source of truth
    response = self._get_response_from_storage(feedback_id)
    if response:
        return response  # Already available (resume case)

    # Setup Pub/Sub listener (optimization for active polling)
    notification_event = threading.Event()
    pubsub_thread = None

    if self.backend == "redis":
        # Start Pub/Sub listener in background thread
        def pubsub_listener():
            try:
                pubsub = self._redis_client.pubsub()
                pubsub.subscribe(f"feedback:{feedback_id}")

                for message in pubsub.listen():
                    if message["type"] == "message":
                        # Notification received
                        notification_event.set()
                        break
            except Exception as e:
                # Pub/Sub failed, no problem (polling fallback)
                print(f"[Pub/Sub] Listener error: {e}")

        pubsub_thread = threading.Thread(target=pubsub_listener, daemon=True)
        pubsub_thread.start()

    # Polling loop (ALWAYS runs, regardless of Pub/Sub)
    poll_interval = 0.5
    elapsed = 0.0

    while elapsed < timeout:
        # Wait for notification OR poll interval
        notification_event.wait(timeout=poll_interval)

        # Always check storage (source of truth)
        response = self._get_response_from_storage(feedback_id)
        if response:
            return response  # Found

        elapsed += poll_interval

    # Timeout
    return None
```

### Key Points

1. **Storage check FIRST**: Resume case immediately finds response
   ```python
   response = self._get_response_from_storage(feedback_id)
   if response:
       return response  # No polling needed
   ```

2. **Pub/Sub is optional**: If it fails, polling still works
   ```python
   try:
       start_pubsub_listener()
   except:
       pass  # No problem, polling works
   ```

3. **Polling ALWAYS runs**: Even with Pub/Sub active
   ```python
   while elapsed < timeout:
       notification_event.wait(poll_interval)  # Wait with timeout
       response = self._get_response_from_storage()  # Always check
   ```

4. **Storage is source of truth**: Pub/Sub only reduces latency
   ```python
   # Pub/Sub notification → Just sets event flag
   # Real data comes from storage
   ```

---

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    Active Polling Phase                       │
│                                                               │
│  Worker Thread 1 (Main):                                      │
│  ┌──────────────────────────────────────────────┐             │
│  │ while elapsed < timeout:                     │             │
│  │     notification_event.wait(0.5s)            │◄────────┐   │
│  │     response = storage.get(feedback_id)      │         │   │
│  │     if response: return response             │         │   │
│  └──────────────────────────────────────────────┘         │   │
│                     ▲                                     │   │
│                     │ Checks storage every 500ms          │   │
│                     │                                     │   │
│  Worker Thread 2 (Background Pub/Sub):                    │   │
│  ┌──────────────────────────────────────────────┐         │   │
│  │ pubsub.subscribe(f"feedback:{id}")           │         │   │
│  │ for message in pubsub.listen():              │         │   │
│  │     if message: notification_event.set() ────┼─────────┘   │
│  └──────────────────────────────────────────────┘             │
│                                                                │
└───────────────────────────────────────────────────────────────┘
                                │
                                │ Timeout (30s)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    Checkpoint Phase                           │
│                                                               │
│  - Worker exits (Pub/Sub subscriber dies)                     │
│  - Response stored in Redis storage (hash)                    │
│  - Pub/Sub message published (but no subscriber → lost)       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                                │
                                │ Resume (later)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    Resume Phase                               │
│                                                               │
│  Worker 2:                                                    │
│  ┌──────────────────────────────────────────────┐             │
│  │ response = storage.get(feedback_id)          │             │
│  │ if response:                                 │             │
│  │     return response  # Found immediately ✅  │             │
│  └──────────────────────────────────────────────┘             │
│                                                               │
│  No polling needed (response already in storage)              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Comparison: Synchronous vs Asynchronous

### Your Terminology Clarification

| Your Term | Meaning | Implementation |
|-----------|---------|----------------|
| **Pubsub (Synchronous)** | Active subscriber listening in real-time | Background thread subscribing to Pub/Sub channel |
| **Polling (Asynchronous)** | Check storage periodically or on-demand | Check Redis hash storage directly |

### Combined Approach

```python
def request_feedback(...):
    # Check storage first (asynchronous/resume case)
    existing = self._get_response_from_storage(feedback_id)
    if existing:
        return existing  # Resume case ✅

    # Start active polling with Pub/Sub optimization (synchronous case)
    response = self._poll_for_response(feedback_id, timeout)
    if response:
        return response  # Active polling ✅

    # Timeout
    raise FeedbackTimeoutException(...)
```

---

## Why This Works

### 1. Storage is Persistent

```redis
# Response stored in Redis Hash (persisted)
HSET feedback:response:deploy_abc123
    feedback_id "deploy_abc123"
    response_type "approval"
    approved "true"
    responded_at "2025-01-28T10:00:00Z"

# Persists even when no subscribers
# Available for days (with TTL)
```

### 2. Pub/Sub is Ephemeral

```redis
# Pub/Sub message (ephemeral)
PUBLISH feedback:deploy_abc123 "completed"

# If no subscribers → Lost immediately
# But doesn't matter because storage has the data
```

### 3. Workflow Behavior

**Active Polling**:
```python
# Pub/Sub: Reduces latency from 500ms → < 100ms
notification_event.wait(0.5)  # Returns early if Pub/Sub notifies
storage.get(feedback_id)      # Still checks storage
```

**Resume After Checkpoint**:
```python
# Pub/Sub: Not relevant (subscriber dead)
storage.get(feedback_id)      # Checks storage directly
# Found immediately, no polling needed
```

---

## Edge Cases Handled

### Edge Case 1: Pub/Sub Fails to Start

```python
try:
    pubsub_thread.start()
except Exception:
    pass  # No problem, polling still works
```

**Result**: Falls back to polling only ✅

### Edge Case 2: Pub/Sub Message Lost

```
Subscriber starts late → Pub/Sub message already published → Lost
```

**Result**: Polling finds it in storage ✅

### Edge Case 3: Storage Check Fails

```python
try:
    response = storage.get(feedback_id)
except Exception:
    # Retry or fail
```

**Result**: This is a real error (source of truth unavailable) ❌

### Edge Case 4: Concurrent Resume

```s
Worker 1: Resumes, checks storage → Found
Worker 2: Resumes, checks storage → Found (idempotent)
```

**Result**: Both workers get response safely ✅

---

## Performance Analysis

### Latency Comparison

| Scenario | Pub/Sub | Polling Only | Hybrid |
|----------|---------|--------------|--------|
| **Active (response within 1s)** | < 100ms | < 500ms | < 100ms ✅ |
| **Active (response within 10s)** | < 100ms | < 500ms | < 100ms ✅ |
| **Resume (response available)** | N/A | 0ms (immediate) | 0ms (immediate) ✅ |

### Resource Usage

| Approach | Redis Connections | Threads | Memory |
|----------|------------------|---------|--------|
| **Pub/Sub Only** | 2 (storage + pubsub) | 1 | Low |
| **Polling Only** | 1 (storage) | 0 | Low |
| **Hybrid** | 2 (storage + pubsub) | 1 | Low |

**Conclusion**: Hybrid has minimal overhead vs Pub/Sub only

---

## Recommendation: Hybrid is Correct

Your understanding is **100% correct**:

✅ **Pubsub (synchronous)**: Optimization for active subscribers
✅ **Polling (asynchronous)**: Reliable fallback for all cases
✅ **Storage as source of truth**: Always check storage

### Final Implementation

```python
class FeedbackManager:
    def _poll_for_response(self, feedback_id, timeout):
        """Hybrid: Pub/Sub optimization + polling reliability."""

        # STEP 1: Check storage immediately (resume case)
        response = self._get_response_from_storage(feedback_id)
        if response:
            return response  # Already available ✅

        # STEP 2: Start Pub/Sub listener (optimization)
        notification_event = threading.Event()
        if self.backend == "redis":
            self._start_pubsub_listener(feedback_id, notification_event)

        # STEP 3: Polling loop (always runs)
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < timeout:
            # Wait for Pub/Sub notification OR poll interval
            notification_event.wait(timeout=poll_interval)

            # Always check storage (source of truth)
            response = self._get_response_from_storage(feedback_id)
            if response:
                return response  # Found ✅

            elapsed += poll_interval

        # STEP 4: Timeout
        return None
```

### Configuration

```python
feedback_config = {
    "notification_mode": "auto",  # Uses hybrid approach
    "poll_interval": 0.5,         # Storage check interval
    "pubsub_enabled": True,       # Enable Pub/Sub optimization (auto for Redis)
}
```

---

## Conclusion

**Your proposal is exactly right**:

| Aspect | Synchronous (Pub/Sub) | Asynchronous (Polling) |
|--------|----------------------|------------------------|
| **Purpose** | Low-latency notification | Reliable data retrieval |
| **Requirement** | Active subscriber | No requirements |
| **Failure mode** | Message lost (OK) | Storage failure (ERROR) |
| **Use case** | Active polling phase | Resume/checkpoint phase |

**Combined**: Best of both worlds ✅

This is the **recommended architecture** for Option 6.

---

**Status**: Design Confirmed
**Next Steps**: Implement hybrid approach in Phase 1
