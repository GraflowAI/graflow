"""Unit tests for graflow.utils.graph.draw_mermaid."""

from __future__ import annotations

import re

from graflow.core.task import Task
from graflow.core.workflow import workflow
from graflow.utils.graph import draw_mermaid


def test_draw_mermaid_nested_parallel_structure() -> None:
    """Ensure draw_mermaid renders nested parallel workflow correctly."""
    with workflow("nested_parallel_mermaid") as wf:
        fetch = Task("fetch")
        transform_a = Task("transform_a")
        subtask_a = Task("subtask_a")
        transform_b = Task("transform_b")
        store = Task("store")

        # Build: fetch >> ((transform_a >> subtask_a) | transform_b) >> store
        fetch >> ((transform_a >> subtask_a) | transform_b) >> store  # type: ignore

        mermaid = draw_mermaid(wf.graph._graph, title="Nested Parallel Workflow")

    assert "graph TD;" in mermaid

    subgraph_match = re.search(r'subgraph\s+(\S+)\["\s*([^"]+)\s*"\]', mermaid)
    assert subgraph_match is not None, mermaid

    parallel_group_id = subgraph_match.group(1)
    parallel_group_label = subgraph_match.group(2).strip()

    assert parallel_group_id.startswith("ParallelGroup")
    assert parallel_group_label.startswith("ParallelGroup")

    lines = {line.strip() for line in mermaid.splitlines() if line.strip()}

    assert any(line.startswith("fetch") and f"--> {parallel_group_id}" in line for line in lines), mermaid

    assert any(line.startswith(parallel_group_id) and "-->" in line and "store" in line for line in lines), mermaid

    assert any(line.startswith("transform_a") and "-->" in line and "subtask_a" in line for line in lines), mermaid

    assert f"{parallel_group_id} --> transform_a;" not in lines
    assert f"{parallel_group_id} --> transform_b;" not in lines
