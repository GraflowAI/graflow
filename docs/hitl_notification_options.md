# HITL Notification Design - Detailed Analysis

**Document**: Notification Options for HITL Feedback
**Date**: 2025-01-28
**Related**: hitl_design.md Option 6

---

## Overview

This document provides detailed analysis of notification mechanisms for HITL feedback responses. When feedback is provided via external API, the workflow needs to be notified so it can resume execution.

---

## Problem Statement

**Challenge**: How should a running workflow (or paused checkpoint) be notified when feedback is provided externally?

**Scenarios**:
1. **Active polling**: Workflow is actively polling for response (within timeout period)
2. **Checkpoint waiting**: Workflow created checkpoint and exited (timeout occurred)
3. **Distributed workers**: Multiple workers may resume the workflow

**Requirements**:
- Low latency for active polling scenarios
- Reliable delivery (no lost notifications)
- Support for distributed environments
- Minimal infrastructure dependencies
- Graceful degradation when notification fails

---

## Option 6A: Redis Pub/Sub Only

### Implementation

```python
class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        # Publish notification via Redis Pub/Sub
        if self.backend == "redis":
            self._redis_client.publish(
                f"feedback:{feedback_id}",
                "completed"
            )
```

```python
# Worker polling loop
def _poll_for_response(self, feedback_id, timeout):
    # Subscribe to Pub/Sub channel
    pubsub = self._redis_client.pubsub()
    pubsub.subscribe(f"feedback:{feedback_id}")

    elapsed = 0
    while elapsed < timeout:
        # Wait for notification with timeout
        message = pubsub.get_message(timeout=0.5)
        if message and message["type"] == "message":
            # Notification received, fetch response
            response = self._get_response(feedback_id)
            if response:
                return response

        elapsed += 0.5

    return None  # Timeout
```

### Pros ✅

1. **Low latency**: Near-instant notification (< 100ms typically)
2. **Built-in**: No extra infrastructure if already using Redis
3. **Scalable**: Supports multiple subscribers (multiple workers)
4. **Simple**: Straightforward pub/sub pattern

### Cons ❌

1. **Redis dependency**: Requires Redis backend
2. **No persistence**: Messages lost if no active subscribers
3. **Fire-and-forget**: No delivery guarantee
4. **Memory backend unsupported**: Only works with Redis backend

### Use Cases

- ✅ Distributed workflows with Redis backend
- ✅ Active polling scenarios (sub-minute timeouts)
- ❌ Long-term checkpoints (hours/days)
- ❌ Memory backend workflows

### Risk Analysis

**Risk**: Notification lost if published when no subscriber listening

**Scenario**:
```
1. Worker polls for feedback (30s timeout)
2. Timeout → Checkpoint created → Worker exits
3. Human provides feedback 5 minutes later
4. Pub/Sub message published → NO SUBSCRIBERS → Lost
5. Worker 2 resumes from checkpoint
6. Polls for response → Finds it in storage ✅
```

**Mitigation**: Response is stored in backend, so workflow finds it on resume. Pub/Sub is optimization for active polling only.

---

## Option 6B: Webhook Only

### Implementation

```python
@dataclass
class FeedbackRequest:
    webhook_url: Optional[str] = None  # Callback URL
    webhook_headers: Optional[dict] = None  # Custom headers
    webhook_retry_count: int = 3  # Retry attempts

class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        # Call webhook if configured
        request = self._get_request(feedback_id)
        if request.webhook_url:
            self._call_webhook(request, response)

    def _call_webhook(self, request, response):
        import requests

        url = request.webhook_url
        headers = request.webhook_headers or {}
        headers["Content-Type"] = "application/json"

        payload = {
            "feedback_id": request.feedback_id,
            "task_id": request.task_id,
            "session_id": request.session_id,
            "response": response.to_dict()
        }

        # Retry logic
        for attempt in range(request.webhook_retry_count):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=5
                )
                if resp.status_code < 300:
                    print(f"[FeedbackManager] Webhook delivered: {url}")
                    return
                else:
                    print(f"[FeedbackManager] Webhook failed: {resp.status_code}")
            except Exception as e:
                print(f"[FeedbackManager] Webhook error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        print(f"[FeedbackManager] Webhook failed after {request.webhook_retry_count} attempts")
```

### Pros ✅

1. **No Redis dependency**: Works with memory backend
2. **External integration**: Notify external systems (Slack, email, etc.)
3. **Flexible**: Can trigger arbitrary actions
4. **Persistent retry**: Built-in retry with exponential backoff

### Cons ❌

1. **Network dependency**: Requires network connectivity
2. **Higher latency**: HTTP overhead (100-1000ms+)
3. **Complexity**: Error handling, retries, security
4. **No broadcast**: Single webhook endpoint (no multi-subscriber)

### Use Cases

- ✅ Integration with external systems (Slack, PagerDuty)
- ✅ Email/SMS notifications
- ✅ Memory backend workflows
- ❌ Low-latency requirements (< 100ms)

### Security Considerations

**Authentication**:
```python
@dataclass
class FeedbackRequest:
    webhook_secret: Optional[str] = None  # HMAC secret

# Generate HMAC signature
import hmac
import hashlib

signature = hmac.new(
    webhook_secret.encode(),
    json.dumps(payload).encode(),
    hashlib.sha256
).hexdigest()

headers["X-Graflow-Signature"] = signature
```

**Webhook receiver validation**:
```python
@app.post("/webhook/feedback")
def receive_feedback_webhook(request: Request):
    # Verify signature
    signature = request.headers.get("X-Graflow-Signature")
    expected = hmac.new(secret, request.body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "Invalid signature")

    # Process webhook
    ...
```

---

## Option 6C: Both Pub/Sub and Webhook

### Implementation

```python
class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        request = self._get_request(feedback_id)

        # Publish to Redis Pub/Sub (if Redis backend)
        if self.backend == "redis":
            self._redis_client.publish(
                f"feedback:{feedback_id}",
                "completed"
            )

        # Call webhook (if configured)
        if request.webhook_url:
            # Call webhook in background thread
            threading.Thread(
                target=self._call_webhook,
                args=(request, response),
                daemon=True
            ).start()
```

### Pros ✅

1. **Best of both worlds**: Low latency + external integration
2. **Redundancy**: Notification via multiple channels
3. **Flexibility**: Use appropriate mechanism per use case
4. **Backward compatible**: Existing Redis users unaffected

### Cons ❌

1. **Complexity**: More code paths, more failure modes
2. **Testing overhead**: Must test both mechanisms
3. **Potential duplication**: Same notification via multiple channels

### Use Cases

- ✅ Enterprise environments with multiple integration points
- ✅ Workflows requiring both low latency and external notifications
- ❌ Simple deployments (overkill)

---

## Option 6D: Server-Sent Events (SSE)

### Implementation

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Global event queue
feedback_events: dict[str, asyncio.Queue] = {}

@app.get("/api/feedback/{feedback_id}/events")
async def feedback_events_stream(feedback_id: str):
    """Server-Sent Events stream for feedback updates."""

    # Create queue for this feedback_id
    queue = asyncio.Queue()
    feedback_events[feedback_id] = queue

    async def event_generator():
        try:
            while True:
                # Wait for event
                event = await queue.get()
                yield f"data: {event}\n\n"
        finally:
            # Cleanup
            del feedback_events[feedback_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# In provide_feedback()
async def provide_feedback(feedback_id, response):
    # Store response
    self._store_response(response)

    # Send SSE event if subscriber exists
    if feedback_id in feedback_events:
        await feedback_events[feedback_id].put("completed")
```

### Pros ✅

1. **Browser-friendly**: Native EventSource API
2. **Persistent connection**: Automatic reconnection
3. **HTTP-based**: Works through proxies/firewalls
4. **No Redis dependency**: Pure HTTP

### Cons ❌

1. **Connection overhead**: One connection per subscriber
2. **Server resources**: Keep-alive connections
3. **Unidirectional**: Server to client only
4. **Not suitable for worker processes**: Designed for web clients

### Use Cases

- ✅ Web UI dashboards
- ✅ Real-time feedback status updates
- ❌ Backend worker processes
- ❌ Distributed workflows

---

## Option 6E: Polling Only (No Notification)

### Implementation

```python
def _poll_for_response(self, feedback_id, timeout):
    """Simple polling without notification."""
    poll_interval = 0.5
    elapsed = 0

    while elapsed < timeout:
        # Check for response
        response = self._get_response(feedback_id)
        if response:
            return response

        # Sleep
        time.sleep(poll_interval)
        elapsed += poll_interval

    return None  # Timeout
```

### Pros ✅

1. **No dependencies**: Works everywhere
2. **Simple**: Minimal code
3. **Reliable**: Direct storage check
4. **Uniform**: Same behavior for all backends

### Cons ❌

1. **Higher latency**: Up to poll_interval (500ms default)
2. **Resource waste**: Continuous polling
3. **No optimization**: Can't reduce latency

### Use Cases

- ✅ Simple deployments
- ✅ Development/testing
- ❌ Production (inefficient)

---

## Comparison Matrix

| Feature | Pub/Sub | Webhook | Both | SSE | Polling |
|---------|---------|---------|------|-----|---------|
| **Latency** | < 100ms | 100-1000ms | < 100ms | < 100ms | < 500ms |
| **Redis Required** | ✅ Yes | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **Network Required** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Multi-subscriber** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Delivery Guarantee** | ❌ No | ⚠️ Retry | ⚠️ Retry | ❌ No | ✅ Yes |
| **External Integration** | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Limited | ❌ No |
| **Complexity** | Low | Medium | High | Medium | Low |
| **Backend Support** | Redis | All | All | All | All |
| **Production Ready** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Web only | ⚠️ Dev only |

---

## Hybrid Approach: Notification + Polling Fallback

### Recommended Architecture

```python
class FeedbackManager:
    def _poll_for_response(self, feedback_id, timeout):
        """Polling with notification optimization."""

        # Setup notification listener (non-blocking)
        notification_received = threading.Event()

        if self.backend == "redis":
            # Subscribe to Pub/Sub in background thread
            def pubsub_listener():
                pubsub = self._redis_client.pubsub()
                pubsub.subscribe(f"feedback:{feedback_id}")

                for message in pubsub.listen():
                    if message["type"] == "message":
                        notification_received.set()
                        break

            thread = threading.Thread(target=pubsub_listener, daemon=True)
            thread.start()

        # Polling loop with notification optimization
        poll_interval = 0.5
        elapsed = 0

        while elapsed < timeout:
            # Check for response
            response = self._get_response(feedback_id)
            if response:
                return response

            # Wait for notification or poll interval
            if notification_received.wait(timeout=poll_interval):
                # Notification received, check immediately
                response = self._get_response(feedback_id)
                if response:
                    return response

            elapsed += poll_interval

        return None  # Timeout
```

### Benefits

- ✅ **Best latency**: Pub/Sub when available, polling fallback
- ✅ **Reliable**: Always polls storage as source of truth
- ✅ **Backend agnostic**: Works with memory or Redis
- ✅ **No lost notifications**: Polling ensures eventual delivery

---

## Recommendations by Use Case

### Simple Deployment (Single Process, Memory Backend)
**Recommendation**: Option 6E (Polling Only)
- No external dependencies
- Simple and reliable
- Acceptable latency for development

### Distributed Workflow (Redis Backend)
**Recommendation**: Option 6A (Redis Pub/Sub) + Polling Fallback (Hybrid)
- Low latency for active polling
- Reliable with polling fallback
- No extra infrastructure

### External System Integration
**Recommendation**: Option 6B (Webhook Only)
- Notify Slack, email, PagerDuty
- Works without Redis
- Retry logic for reliability

### Enterprise Environment
**Recommendation**: Option 6C (Both Pub/Sub + Webhook)
- Low latency for workers (Pub/Sub)
- External notifications (Webhook)
- Maximum flexibility

### Web Dashboard
**Recommendation**: Option 6D (SSE) for UI + Option 6A (Pub/Sub) for workers
- Real-time UI updates
- Low latency worker notifications

---

## Implementation Phases

### Phase 1: Core (Polling + Pub/Sub)
```python
class FeedbackManager:
    def __init__(self, backend, notification_mode="auto"):
        """
        notification_mode:
            "auto" - Pub/Sub if Redis, else polling only
            "polling" - Polling only (no notifications)
            "pubsub" - Redis Pub/Sub only (requires Redis)
        """
```

**Priority**: High
**Timeline**: Week 1-2
**Deliverables**:
- Polling loop implementation
- Redis Pub/Sub integration
- Hybrid approach (Pub/Sub + polling fallback)

### Phase 2: Webhook Support
```python
@dataclass
class FeedbackRequest:
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_retry_count: int = 3
```

**Priority**: Medium
**Timeline**: Week 3-4
**Deliverables**:
- Webhook calling logic
- Retry with exponential backoff
- HMAC signature verification
- Error handling and logging

### Phase 3: SSE for Web UI
```python
@app.get("/api/feedback/{feedback_id}/events")
async def feedback_events_stream(feedback_id: str):
    # SSE implementation
```

**Priority**: Low (Optional)
**Timeline**: Week 5+
**Deliverables**:
- SSE endpoint
- Browser EventSource integration
- Automatic reconnection

---

## Recommended Decision

For **Option 6**, recommend:

**Phase 1 Implementation**: **Hybrid (Pub/Sub + Polling Fallback)**
```python
notification_mode = "auto"  # Default
# - Uses Pub/Sub if Redis backend
# - Falls back to polling for Memory backend
# - Always polls storage as source of truth
```

**Configuration**:
```python
context = ExecutionContext.create(
    feedback_backend="redis",
    feedback_config={
        "notification_mode": "auto",  # auto, polling, pubsub, webhook
        "poll_interval": 0.5,
        "webhook_url": None,  # Optional
    }
)
```

**Rationale**:
1. ✅ Works for all backends (Memory, Redis)
2. ✅ Low latency when Pub/Sub available
3. ✅ Reliable with polling fallback
4. ✅ Simple to implement (Phase 1)
5. ✅ Extensible (webhook in Phase 2)

---

## Open Questions for Discussion

1. **Webhook retry strategy**: Exponential backoff or fixed interval?
2. **Webhook timeout**: How long to wait for webhook response? (default: 5s)
3. **Notification failure handling**: Log only or raise exception?
4. **Multi-webhook**: Support multiple webhook URLs per feedback?
5. **Notification persistence**: Store notification delivery status?

---

**Status**: Discussion Draft
**Next Steps**: Finalize notification mode choice
