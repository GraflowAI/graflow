# Queue Architecture Simplification Proposal

**Author**: Graflow Team  
**Date**: 2025-10-23  
**Status**: Draft (discussion)

---

## Background

Graflow currently supports multiple task queue backends through `TaskQueueFactory`:

- `ExecutionContext` instantiates either `InMemoryTaskQueue` or `RedisTaskQueue` based on configuration.
- `TaskWorker` accepts any `TaskQueue` implementation, and the Redis-backed queue is typically used for distributed workers.
- `RedisCoordinator` (used by `ParallelGroup` when the Redis backend is selected) also accepts a `RedisTaskQueue`, which today can be the same instance that an `ExecutionContext` uses.

This flexibility adds complexity:

- `ExecutionContext` serialization and checkpoint logic must handle Redis-specific state even when workflows primarily run in-process.
- Tests and examples need to account for queue backend permutations.
- The separation between “engine-driven in-process execution” and “coordinated distributed execution” is blurred, making it harder to reason about queue ownership and lifecycle.

---

## Proposal Overview

Simplify queue responsibilities by **specialising backends**:

1. **ExecutionContext always uses `InMemoryTaskQueue`.**
   - Remove `queue_backend` from `ExecutionContext.create`.
   - `TaskQueueFactory` is still available for other components but no longer used by `ExecutionContext`.
   - All in-process and checkpoint scenarios rely on a predictable, serialisable queue.

2. **`RedisTaskQueue` is dedicated to distributed coordination.**
   - `TaskWorker` will only accept a `RedisTaskQueue` (local helper to instantiate from Redis config).
   - `RedisCoordinator` obtains its own `RedisTaskQueue` instance independent of the execution context queue.
   - `ParallelGroup` execution via `RedisCoordinator` publishes units of work to Redis queues consumed exclusively by TaskWorkers.

This yields a clean separation:  
`ExecutionContext` ⇢ local runtime with in-memory queue,  
`RedisCoordinator` + `TaskWorker` ⇢ distributed execution path with Redis queue.

---

## Detailed Changes

### ExecutionContext / Engine

- Remove the `queue_backend` argument and related wiring from:
  - `ExecutionContext.__init__`
  - `ExecutionContext.create`
  - `ExecutionContext.create_branch_context`
- Instantiate `InMemoryTaskQueue` directly with the start node.
- Update checkpoint serialization to drop Redis queue assumptions.

### TaskQueueFactory

- Keep for external use, but mark Redis branch as “for coordinator/workers”.
- Consider `TaskQueueFactory` helpers that return in-memory queues for tests.

### Redis Task Flow

- `RedisCoordinator` constructs its own `RedisTaskQueue` using supplied Redis client + prefix.
- `TaskWorker` initialisation is simplified:
  - Accept Redis connection details instead of an arbitrary `TaskQueue`.
  - Internally create `RedisTaskQueue` and manage its lifecycle.

### Executor (`graflow/coordination/executor.py`)

- Ensure that when a `ParallelGroup` selects the Redis backend, the executor passes the proper Redis configuration to both `RedisCoordinator` and worker processes.
- ExecutionContext’s queue (`InMemoryTaskQueue`) is used only for scheduling tasks until they are handed off to the coordinator.

---

## Impacts

| Area | Impact |
|------|--------|
| **API** | `ExecutionContext.create` signature change (no `queue_backend` argument). |
| **Workers** | TaskWorker CLI / constructor must shift to Redis-only and accept Redis config inputs. |
| **Testing** | Simplifies most tests (only in-memory queues), but distributed tests need explicit Redis setup through coordinator/worker APIs. |
| **Documentation** | Update queue/worker docs to reflect new separation. |
| **Checkpoints** | Simplifies checkpoint state (always in-memory queue). Need migration notes for existing checkpoints that stored Redis queue metadata. |

---

## Migration Plan

1. **Phase 1 – Code Refactor**
   - Implement ExecutionContext change.
   - Adjust TaskWorker/RedisCoordinator/executor wiring.
   - Update tests to work with the new defaults.
2. **Phase 2 – Documentation**
   - Refresh worker guides, distributed execution docs, and feature comparison.
3. **Phase 3 – Cleanup**
   - Remove unused Redis queue serialization helpers from ExecutionContext.
   - Monitor for user feedback; consider providing convert script for legacy checkpoints if needed.

---

## Open Questions

- Keep TaskWorker in `graflow.worker.worker` so it remains an independent process/service entrypoint.
- For distributed queues we focus on Redis-compatible services (e.g., Valkey); supporting non-Redis backends like SQS is out of scope.
- Provide configuration hooks so users can supply either a Redis URI or discrete host/port settings; enforcing consistency across coordinator and TaskWorker is left to deployment configuration.

---

## Next Steps

1. Review proposal with core maintainers.
2. Finalise API changes (`ExecutionContext.create`, TaskWorker initialisation).
3. Implement Phase 1 refactor behind feature flag or short-lived branch.
