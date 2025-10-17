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

2. **Phase 1 – API modernization** ✅
   - ✅ `ParallelGroup.__init__` no longer registers membership edges in the runtime graph.
   - ✅ Added `TaskGraph.get_parallel_group_members()` to expose container membership from node metadata.
   - ✅ Updated `_has_successors`, merge operators, and unit tests (`tests/core/test_parallel_group.py`, `tests/core/test_sequential_task.py`) to call the new API instead of relying on `graph.successors(group)`.
   - ✅ Extended graph tests to assert both membership sets and external successor edges.

3. **Phase 2 – Visualization refresh** ✅
   - ✅ `build_graph()` now synthesizes container edges from `ParallelGroup.tasks`, ensuring visualization helpers include member nodes without mutating the execution graph.
   - ✅ Mermaid/DOT utilities reuse the explicit membership lists when classifying internal vs external edges (see `graflow/utils/graph.py`).
   - ✅ Regression coverage: `tests/test_graph.py` and `tests/utils/test_graph_mermaid.py` confirm container rendering without relying on implicit successor edges.

4. **Phase 3 – Execution + Queue strategy** ✅
   - ✅ Introduced `ExecutionContext.create_branch_context()` / `merge_results()` / `mark_branch_completed()` to isolate branch queues while sharing read-only graph metadata.
   - ✅ Branch contexts copy parent channel state on creation and merge results back after successful completion.
   - ✅ `ThreadingCoordinator.execute_group()` now spins up branch contexts per task, executes them with dedicated `WorkflowEngine` instances, and merges outcomes before resuming the parent queue.
   - 📎 Additional design details remain documented in `docs/parallel_execution_queue_design.md`.

5. **Phase 4 – Cleanup** ✅
   - ✅ Regression suite updated (`tests/test_graph.py`, `tests/core/test_parallel_group.py`, `tests/core/test_sequential_task.py`, `tests/utils/test_graph_mermaid.py`).
   - ✅ Documentation refreshed to describe membership APIs and branch-context execution flow.

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
