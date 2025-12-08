# Runtime Graph Parent Linking (Codex)

## Observed Issue
- Runtime graph nodes are disconnected for normal workflow execution; parent-child edges rarely appear.
- `Tracer.on_task_start` derives `parent_task_id` from `context.current_task_id` before the task is pushed to the stack (`graflow/trace/base.py:299-323`, `graflow/core/context.py:1193-1226`). The stack is empty for most tasks, so `parent_name=None` and no edge is added.
- Successors are enqueued **after** leaving `context.executing_task`, which pops the stack. `add_to_queue` records `parent_span_id` from `current_task_id`, but by then it is `None` (`graflow/core/context.py:693-711`).
- The queue drops parent metadata on dequeue: `TaskQueue.get_next_task` returns only a task_id, not the `TaskSpec` that carries `parent_span_id` (`graflow/queue/base.py:123-126`). Even when a parent_span_id is set (e.g., dynamic tasks enqueued inside a handler), `on_task_start` cannot see it.
- Net effect: `_runtime_graph` never receives parent information during normal engine runs.

## Recommendations
1) **Preserve parent when enqueueing successors**  
   - Keep the current task on the stack (or stash its id) while successors are queued. Options: move successor scheduling inside the `with context.executing_task(task)` block, or capture a temporary `current_parent_task_id` before popping and let `add_to_queue` use that when the stack is empty.

2) **Propagate parent metadata to tracer on start**  
   - Stop discarding `TaskSpec`. Dequeue a `TaskSpec` (or return both task_id and parent_span_id) and feed `parent_span_id` into `executing_task`/`on_task_start` when `current_task_id` is missing. Clear the cached parent after consumption to avoid bleed-over.

3) **Distributed alignment**  
   - If sticking with worker-side `attach_to_trace(parent_span_id=...)`, persist that parent in the tracer and treat it as a fallback for the first task in the process. This complements (not replaces) the queue-based parent propagation.

4) **Regression coverage**  
   - Add tests that execute a simple two-node graph via `WorkflowEngine.execute` and assert the runtime graph has an edge parent→child.  
   - Add a dynamic-task case (child enqueued inside handler) and a distributed worker attach case to confirm edges are emitted when parent data comes from `TaskSpec` or `attach_to_trace`.

## Suggested Fix Path
- Adjust `WorkflowEngine/ExecutionContext` to operate on `TaskSpec` when dequeuing and to pass `parent_span_id` to tracer hooks.  
- Ensure successor enqueue happens while the parent id is still available (stack or cached).  
- Update tracer start logic to prefer stack parents, then queued `parent_span_id`, then worker `parent_span_id` fallback.  
- Re-run runtime-graph tests and add new assertions for edges once the propagation is wired.
