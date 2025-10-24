# TaskResolver Decoupling Plan

**Author:** Graflow Team  
**Date:** 2024-06-25  
**Status:** Draft (design exploration)

---

## Motivation

`TaskResolver` instances are currently owned by each `ExecutionContext`. While this keeps resolution state local to a workflow run, it also creates tight coupling:

- `TaskSpec` and queue implementations must always carry a reference back to an `ExecutionContext` to serialize/resolve tasks.
- Distributed workers and CLI tooling must construct dummy contexts solely to satisfy resolver usage.
- Registering globally-shared tasks requires hopping through an active execution context, even when the task definitions are process-wide.

We want the resolver concept to be reusable across queues and workers without requiring an `ExecutionContext`.

---

## Goals

1. Make `TaskResolver` a reusable, context-agnostic component.
2. Allow both local contexts and distributed workers to share resolvers via a registry.
3. Preserve existing serialization strategies (`reference`, `pickle`, `source`) while simplifying call sites.

Non-goals: introduce backwards compatibility layers or incremental adapters—the refactor can be breaking.

---

## Current State

```
ExecutionContext
  └─ _task_resolver: TaskResolver
TaskSpec.task_data -> execution_context.task_resolver.serialize_task(...)
TaskSpec.get_task() -> execution_context.task_resolver.resolve_task(...)
```

As a result, queues and workers must depend on an `ExecutionContext` even when they only need serialization facilities.

---

## Proposed Architecture

### 1. Standalone TaskResolver

Turn `TaskResolver` into a self-sufficient object that can be instantiated and registered independently of contexts. Provide helpers to:

- register callables (`register_task(name, task)`)
- serialize / resolve without referencing `ExecutionContext`

### 2. Global Resolver Registry

Introduce a simple registry:

```python
class TaskResolverRegistry:
    def register(self, resolver_id: str, resolver: TaskResolver) -> None: ...
    def get(self, resolver_id: str = "default") -> TaskResolver: ...
```

Execution contexts register their preferred resolver on creation, but distributed workers can also register resolvers up front (e.g., via CLI configuration).

### 3. TaskSpec Refactor

`TaskSpec` maintains either a direct executable or a serialized envelope plus `resolver_id`. Serialization flows:

```python
spec = TaskSpec(
    executable=my_task,
    resolver_id="etl",
)
spec.ensure_serialized()  # uses registry.get("etl")
queue.enqueue(spec)
```

During dequeue:

```python
spec = queue.dequeue()
spec.ensure_deserialized()
task = spec.executable
```

### 4. ExecutionContext Integration

Contexts no longer own resolvers; instead, they accept a `resolver_id` (default `"default"`). They can still instantiate a resolver for local use, but the registry becomes the shared source of truth.

### 5. Queue Implementations

`InMemoryTaskQueue` can continue to work with in-process executables and only fall back to serialization when needed.

`RedisTaskQueue` receives the registry + resolver ID via constructor, removing the need to touch `ExecutionContext`.

### 6. Worker CLI

Workers accept a resolver identifier and register required tasks at startup. Example CLI options:

```
--resolver-id default
--resolver-module examples.tasks
```

Workers load the modules, register tasks with the registry, and then start polling.

---

## Migration Steps

1. Implement `TaskResolverRegistry` and expose a `default_registry`.
2. Update `TaskResolver` APIs to work without `ExecutionContext`.
3. Refactor `TaskSpec` to use `resolver_id` + serialized payload instead of accessing `execution_context`.
4. Update queue implementations, `ExecutionContext`, and workers to pass resolver identifiers explicitly.
5. Refresh distributed examples/tests to showcase resolver registration on both producer and worker sides.

---

## Open Questions

- How do we register tasks defined at runtime (dynamic workflows) so that workers can resolve them?
- Should resolver registration be declarative (config file) or programmatic (executing code at startup)?
- What is the story for pickled tasks requiring custom imports or dependencies on the worker side?

---

## Next Steps

1. Prototype the registry and ensure it covers existing serialization modes.
2. Implement the `TaskSpec` refactor and adjust queues accordingly.
3. Update worker CLI to support resolver registration and test end-to-end.
