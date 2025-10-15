---
title: ParallelGroup Graph Refactor Plan
status: proposal
created: 2024-10-14
authors:
  - Codex Assistant
---

# ParallelGroup Graph Refactor Plan

## Background

Current ParallelGroup nodes register graph edges to each member task during `ParallelGroup.__init__`. This design lets tests and visualization utilities rely on `graph.successors(group_id)` returning both container members and true downstream nodes. However, the same edges lead to duplicate scheduling: after a parallel group runs, `WorkflowEngine.execute()` walks `graph.successors(group_id)` and queues every successor, including internal members that were already executed by `GroupExecutor`. Under nested or multi-branch flows, this re-queues inner tasks, creating double execution (see `fetch >> ((transform_a >> subtask_a) | transform_b) >> store`).

## Goal

Refactor graph modeling so that:

- ParallelGroup membership is tracked outside the raw successor list.  
- Execution scheduling does not re-enqueue group members after the group finishes.  
- Graph visualizations and merge rules remain coherent without relying on implicit internal edges.  
- Queue isolation design (`docs/parallel_execution_queue_design.md`) can be layered in without conflicting assumptions.

## High-Level Approach

1. **Phase 0 – Stabilize execution** ✅
   - ✅ Implemented in `graflow/core/engine.py`: `WorkflowEngine.execute()` now filters out internal member IDs before enqueuing successors, preventing duplicate runs while keeping the current graph shape intact.
     ```python
     # graflow/core/engine.py
     successors = list(graph.successors(task_id))
     if isinstance(task, ParallelGroup):
         member_ids = {member.task_id for member in task.tasks}
         successors = [succ for succ in successors if succ not in member_ids]
     for succ in successors:
         succ_task = graph.get_node(succ)
         context.add_to_queue(succ_task)
     ```
   - Regression test: `tests/core/test_sequential_task.py::test_nested_parallel_execution` should confirm execution order `["fetch", "transform_a", "subtask_a", "transform_b", "store"]` with no duplicates.

2. **Phase 1 – API modernization**
   - Stop registering parallel membership edges in `ParallelGroup.__init__`.
     ```python
     class ParallelGroup(Executable):
         def __init__(self, tasks: list[Executable]) -> None:
             super().__init__()
             self._task_id = self._get_group_name()
             self.tasks = list(tasks)
             self._register_to_context()
             # removed: for task in self.tasks: self._add_dependency_edge(self._task_id, task.task_id)
     ```
   - Add `TaskGraph.get_parallel_group_members(group_id)` (or similar) for explicit membership access.
     ```python
     class TaskGraph:
         def get_parallel_group_members(self, group_id: str) -> list[str]:
             node_data = self._graph.nodes[group_id]
             task = node_data.get("task")
             if isinstance(task, ParallelGroup):
                 return [member.task_id for member in task.tasks]
             return []
     ```
   - Update `ParallelGroup._has_successors`, merge logic, and tests to use the new API instead of decoding successors.
     ```python
     def _has_successors(self) -> bool:
         graph = current_workflow_context().graph._graph
         successors = list(graph.successors(self.task_id))
         member_ids = {task.task_id for task in self.tasks}
         return any(succ for succ in successors if succ not in member_ids)
     ```
     Test updates:
     ```python
     members = ctx.graph.get_parallel_group_members("group1")
     assert set(members) == {"task1", "task2"}
     ```

3. **Phase 2 – Visualization refresh**
   - Rewrite Mermaid/DOT transforms in `graflow/utils/graph.py` to rely on `ParallelGroup.tasks` rather than graph successors.
     ```python
     parallel_members = [member.task_id for member in task.tasks]
     for member_id in parallel_members:
         # render inside subgraph / cluster
     external_successors = [
         succ for succ in graph.successors(group_id)
         if succ not in parallel_members
     ]
     ```
   - Preserve container rendering (double boxes, subgraphs) by iterating over member lists and scanning downstream edges separately.

4. **Phase 3 – Execution + Queue strategy**
   - Implement branch-specific execution contexts (`ExecutionContext.create_branch_context`) that clone shared graph/channel but spawn isolated task queues (reusing session ID scheme).  
     ```python
    def create_branch_context(self, task_id: str) -> "ExecutionContext":
        branch_session_id = f"{self.session_id}_{task_id}"
        return ExecutionContext(
            graph=self.graph,
            start_node=task_id,   # ensure the branch queue seeds with the target task
            session_id=branch_session_id,
            parent_context=self,
            queue_backend=self._queue_backend_type,
            channel_backend=self._channel_backend_type,
            max_steps=self.max_steps,
            default_max_retries=self.default_max_retries,
            config=self._original_config,
        )
     ```
   - Ensure `GroupExecutor` / `ThreadingCoordinator` use the branch contexts so main queue only tracks external successors.
     ```python
     branch_ctx = execution_context.create_branch_context(task.task_id)
     engine.execute(branch_ctx, start_task_id=task.task_id)
     ```
   - Details on queue isolation live in `docs/parallel_execution_queue_design.md`; both documents assume branch contexts keep read-only structures but create new queue/channel instances. After branch execution, coordinators call `merge_results` and `mark_branch_completed` to fold sub-context state back into the parent session.

5. **Phase 4 – Cleanup**
   - Remove compatibility helpers; ensure all tests (`tests/core/test_parallel_group.py`, `tests/core/test_sequential_task.py`, `tests/utils/test_graph_mermaid.py`, `tests/scenario/test_parallel_diamond.py`) cover the new model.
   - Update developer docs and code comments to reflect the container-first design.

## Execution Notes

- Refactoring will touch multiple subsystems (core task graph, engine, coordinators, visualization). Change sequencing is important—finish one phase with green tests before continuing.
- For Phase 1, tests that currently assert `graph.successors(group)` contain members must be rewritten to call the new API. Removing implicit knowledge from tests will expose reliance in other modules.
- Mermaid/DOT generation should be validated manually (e.g., rerun `test_nested_mermaid.py`, `tests/utils/test_graph_mermaid.py`) after transform changes; confirm container subgraphs still highlight members without relying on successor edges.
- Queue isolation (Phase 3) should be validated against `tests/scenario/test_parallel_diamond.py` to prove race fixes remain intact after removing member edges, aligning with the `parallel_execution_queue_design.md` branch-context approach.

## Risks & Mitigations

- **Breaking API consumers**: Developers using `graph.successors(group)` will see behavior changes. Mitigation: document the new API prominently and provide migration guidance in release notes.  
- **Visualization regressions**: Without careful updates, container diagrams may lose detail. Mitigation: add regression snapshots or assertions to `tests/utils/test_graph_mermaid.py`.
- **Execution semantics**: Nested group interactions (e.g., group inside group) must still honor ordering constraints. Mitigation: expand `tests/core/test_sequential_task.py` to cover nested groups after refactor.

## Next Steps

1. Land Phase 0 patch (successor filtering) if not already committed.  
2. Schedule Phase 1 refactor work and update unit tests accordingly.  
3. Iterate through phases with targeted test suites after each milestone.  
4. Publish migration notes once the new model stabilizes.
