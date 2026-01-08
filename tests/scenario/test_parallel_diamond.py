"""
Regression tests for parallel diamond workflows.
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def test_parallel_diamond_runs_store_once_after_parallel_transforms():
    """
    Ensure fan-out/fan-in pattern does not execute the sink prematurely or twice.

    Reproduces the behaviour of examples/02_workflows/operators_demo.py::example_3_mixed.
    """
    execution_log: list[str] = []

    with workflow("diamond_parallel") as ctx:

        @task("fetch")
        def fetch():
            execution_log.append("fetch")
            return "fetch_complete"

        @task("transform_a")
        def transform_a():
            execution_log.append("transform_a")
            return "transform_a_complete"

        @task("transform_b")
        def transform_b():
            execution_log.append("transform_b")
            return "transform_b_complete"

        @task("store")
        def store():
            execution_log.append("store")
            return "store_complete"

        # Build diamond: fetch -> (transform_a | transform_b) -> store
        fetch >> (transform_a | transform_b) >> store  # type: ignore[arg-type]

        print(f"Graph:\n {ctx.graph}", flush=True)

        ctx.execute("fetch")

    store_indices = [idx for idx, label in enumerate(execution_log) if label == "store"]
    assert len(store_indices) == 1, f"'store' executed {len(store_indices)} times: {execution_log}"

    store_idx = store_indices[0]
    for label in ("transform_a", "transform_b"):
        assert label in execution_log, f"Missing execution for {label}: {execution_log}"
        assert store_idx > execution_log.index(label), f"'store' ran before {label}: {execution_log}"
