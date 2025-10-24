# Redis Task Queue Decoupling Plan

**Author:** Graflow Team  
**Date:** 2024-06-25  
**Status:** Draft (design exploration)

---

## Motivation

`RedisTaskQueue` currently requires an `ExecutionContext` instance. The queue uses the context to:

- Serialize tasks via `TaskSpec.task_data`, which delegates to `ExecutionContext.task_resolver`.
- Rehydrate tasks during `dequeue()` by resolving serialized payloads back into callables.
- Retain a `session_id`, even though Redis queue keys have already moved to prefix-only namespaces.

For distributed workers, coupling the queue to an in-process `ExecutionContext` is awkward:

- Dedicated workers do not own the workflow graph or resolver state of the original runner.
- Dummy contexts must be created purely to satisfy constructor requirements.
- Sharing queues across processes requires passing the original session metadata around.

Decoupling Redis queues from `ExecutionContext` lets workers attach to queues with only Redis connection details and a task resolver registry.

---

## Goals

1. Allow `RedisTaskQueue` to operate without an `ExecutionContext`.
2. Preserve the ability to serialize/deserialize task callables for distributed execution.
3. Keep the public API changes minimal and well-documented.

---

## Current Architecture (simplified)

```
ExecutionContext
  └─ TaskResolver (serialize/resolve)
  └─ RedisTaskQueue(execution_context)
        ├─ enqueue(TaskSpec(execution_context))
        └─ dequeue() -> TaskSpec(execution_context)
```

`TaskSpec` stores a direct reference to the originating context. As a result, every queue operation depends on the context.

---

## Proposed Architecture

### 1. Task Serialization Envelope

Introduce a lightweight `SerializedTask` data class that contains everything required to rehydrate a task:

```python
@dataclass
class SerializedTask:
    strategy: str
    payload: dict[str, Any]
```

`TaskResolver.serialize()` and `.resolve()` will accept/return this envelope instead of working through `ExecutionContext`.

### 2. TaskSpec Without ExecutionContext

Refactor `TaskSpec` so that it optionally holds serialized task data instead of an entire context:

```python
class TaskSpec:
    executable: Executable | None
    serialized_task: SerializedTask | None
    resolver_id: str  # identifier for lookup
```

When enqueuing locally, `executable` can be stored directly. When enqueuing for Redis, `serialized_task` is populated and `executable` cleared.

### 3. Resolver Registry

Create a `TaskResolverRegistry` singleton (or dependency-injected mapping) that associates `resolver_id` with concrete resolver instances. Both the producer and worker processes register their resolvers under well-known IDs (e.g., `"default"`).

`RedisTaskQueue` will accept:

```python
RedisTaskQueue(
    redis_client,
    key_prefix,
    resolver_registry: TaskResolverRegistry,
    default_resolver_id: str = "default",
)
```

### 4. RedisTaskQueue Responsibilities

- `enqueue(TaskSpec)`:
  - Ensure `TaskSpec.serialized_task` is populated (using resolver registry when needed).
  - Store serialized envelope + metadata in Redis.

- `dequeue()`:
  - Retrieve envelope.
  - Resolve to an executable using registry + `resolver_id`.
  - Return a `TaskSpec` ready for `TaskWorker`.

### 5. ExecutionContext Changes

In-memory queues remain ExecutionContext-bound. For Redis handoff:

- When creating a `TaskSpec` destined for Redis, call a helper to produce a serialized-only spec.
- No longer pass `ExecutionContext` into `RedisTaskQueue`.

### 6. TaskWorker Changes

Workers will be constructed with:

```python
TaskWorker(
    queue=RedisTaskQueue(...),
    resolver_registry=TaskResolverRegistry.global(),
    resolver_id="default",
)
```

If needed, the worker may accept a mapping of resolver IDs to modules to register at startup.

---

## Data Format Example

Stored JSON per task:

```json
{
  "task_id": "extract_data",
  "status": "ready",
  "created_at": 1719300000.0,
  "strategy": "reference",
  "resolver_id": "default",
  "serialized_task": {
    "strategy": "reference",
    "payload": {
      "module": "examples.tasks",
      "name": "extract_data"
    }
  },
  "group_id": "etl_batch"
}
```

---

## Migration Plan

1. Implement `SerializedTask` and `TaskResolverRegistry`.
2. Refactor `TaskSpec` to support resolver-based serialization.
3. Update `RedisTaskQueue` APIs and adjust all call sites (coordinator, worker, tests, examples).
4. Document new worker CLI options (resolver registration strategy).

---

## Open Questions

- Should resolver registration be global or injected per queue?
- How do we handle custom serialization strategies (pickle/source) across processes?
- Do we need schema versioning for queue payloads?

---

## Next Steps

1. Prototype `SerializedTask` + registry.
2. Refactor enqueue/dequeue logic in `RedisTaskQueue`.
3. Update distributed examples and worker CLI to register resolvers explicitly.
4. Write migration instructions for existing deployments.
