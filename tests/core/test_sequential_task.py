"""Pytest-style tests for SequentialTask functionality."""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.core.task import ParallelGroup, SequentialTask, Task, TaskWrapper, chain
from graflow.core.workflow import WorkflowContext


@pytest.fixture
def execution_context():
    """Create execution context for tests."""
    graph = TaskGraph()
    return ExecutionContext.create(graph, start_node=None, max_steps=10)


class TestSequentialTaskBasics:
    """Test cases for SequentialTask basic functionality."""

    def test_sequential_task_creation(self):
        """Test SequentialTask creation and basic properties."""
        with WorkflowContext("test_workflow"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            # Create sequential task: a >> b >> c
            result = task_a >> task_b >> task_c

            assert isinstance(result, SequentialTask)
            assert len(result.tasks) == 3
            assert result.leftmost == task_a
            assert result.rightmost == task_c
            assert result.tasks[0] == task_a
            assert result.tasks[1] == task_b
            assert result.tasks[2] == task_c

    def test_chain_helper_creates_dependencies(self):
        """chain helper should mirror >> semantics and add graph edges."""
        with WorkflowContext("test_workflow") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            chained = chain(task_a, task_b, task_c)

            assert isinstance(chained, SequentialTask)
            assert [t.task_id for t in chained.tasks] == ["task_a", "task_b", "task_c"]

            graph = ctx.graph._graph
            assert ("task_a", "task_b") in graph.edges()
            assert ("task_b", "task_c") in graph.edges()

    def test_sequential_task_leftmost_property(self):
        """Test leftmost property returns first task in chain."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")

            sequential = task_a >> task_b

            assert sequential.leftmost == task_a
            assert sequential.leftmost.task_id == "task_a"

    def test_sequential_task_rightmost_property(self):
        """Test rightmost property returns last task in chain."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            sequential = task_a >> task_b >> task_c

            assert sequential.rightmost == task_c
            assert sequential.rightmost.task_id == "task_c"

    def test_sequential_task_task_id(self):
        """Test task_id returns leftmost task_id."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")

            sequential = task_a >> task_b

            assert sequential.task_id == "task_a"

    def test_sequential_task_iteration(self):
        """Test iteration over tasks in the chain."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            sequential = task_a >> task_b >> task_c

            task_ids = [task.task_id for task in sequential]
            assert task_ids == ["task_a", "task_b", "task_c"]

    def test_sequential_task_indexing(self):
        """Test indexing into the task chain."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            sequential = task_a >> task_b >> task_c

            assert sequential[0] == task_a
            assert sequential[1] == task_b
            assert sequential[2] == task_c

    def test_sequential_task_repr(self):
        """Test string representation of SequentialTask."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")

            sequential = task_a >> task_b

            repr_str = repr(sequential)
            assert "SequentialTask" in repr_str
            assert "task_a >> task_b" in repr_str


class TestSequentialTaskOperators:
    """Test cases for SequentialTask operators."""

    def test_rshift_operator_simple(self):
        """Test >> operator creates SequentialTask."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")

            result = task_a >> task_b

            assert isinstance(result, SequentialTask)
            assert result.leftmost == task_a
            assert result.rightmost == task_b

    def test_rshift_operator_chaining(self):
        """Test >> operator chaining: (a >> b) >> c."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            result = task_a >> task_b >> task_c

            assert isinstance(result, SequentialTask)
            assert len(result.tasks) == 3
            assert result.leftmost == task_a
            assert result.rightmost == task_c

    def test_rshift_operator_with_sequential_task(self):
        """Test task >> SequentialTask creates combined chain."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            # Create: a >> (b >> c)
            chain_bc = task_b >> task_c
            result = task_a >> chain_bc

            assert isinstance(result, SequentialTask)
            assert len(result.tasks) == 3
            assert result.leftmost == task_a
            assert result.rightmost == task_c

    def test_lshift_operator_simple(self):
        """Test << operator creates SequentialTask."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")

            # b << a means a >> b
            result = task_b << task_a

            assert isinstance(result, SequentialTask)
            assert result.leftmost == task_a
            assert result.rightmost == task_b

    def test_or_operator_with_sequential_task(self):
        """Test | operator with SequentialTask uses leftmost task."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            # (a >> b) | c should create ParallelGroup([a, c])
            chain_ab = task_a >> task_b
            result = chain_ab | task_c

            assert isinstance(result, ParallelGroup)
            assert len(result.tasks) == 2
            # Check that leftmost (a) is used, not rightmost (b)
            assert task_a in result.tasks
            assert task_c in result.tasks
            assert task_b not in result.tasks

    def test_or_operator_two_sequential_tasks(self):
        """Test | operator between two SequentialTasks."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")
            task_d = Task("task_d")

            # (a >> b) | (c >> d) should create ParallelGroup([a, c])
            chain_ab = task_a >> task_b
            chain_cd = task_c >> task_d
            result = chain_ab | chain_cd

            assert isinstance(result, ParallelGroup)
            assert len(result.tasks) == 2
            assert task_a in result.tasks
            assert task_c in result.tasks


class TestSequentialTaskGraphStructure:
    """Test cases for SequentialTask graph structure."""

    def test_simple_chain_graph_edges(self):
        """Test graph edges for simple chain: a >> b >> c."""
        with WorkflowContext("test") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            _ = task_a >> task_b >> task_c

            graph = ctx.graph._graph

            # Verify edges: a -> b, b -> c
            assert ("task_a", "task_b") in graph.edges()
            assert ("task_b", "task_c") in graph.edges()

    def test_parallel_with_sequential_graph_edges(self):
        """Test graph edges for (a >> b) | c."""
        with WorkflowContext("test") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")

            _ = (task_a >> task_b) | task_c

            graph = ctx.graph._graph

            # Find the parallel group
            parallel_groups = [n for n in graph.nodes() if n.startswith("ParallelGroup_")]
            assert len(parallel_groups) == 1
            group_id = parallel_groups[0]

            # Verify membership via TaskGraph API
            members = ctx.graph.get_parallel_group_members(group_id)
            assert set(members) == {"task_a", "task_c"}

            # Verify group does not directly connect to members in execution graph
            group_successors = list(graph.successors(group_id))
            assert group_successors == []

            task_a_successors = list(graph.successors("task_a"))
            assert "task_b" in task_a_successors

    def test_nested_parallel_graph_structure(self):
        """Test graph structure for: fetch >> ((transform_a >> subtask_a) | transform_b) >> store."""
        with WorkflowContext("test") as ctx:
            fetch = Task("fetch")
            transform_a = Task("transform_a")
            subtask_a = Task("subtask_a")
            transform_b = Task("transform_b")
            store = Task("store")

            # Build the workflow
            _ = fetch >> ((transform_a >> subtask_a) | transform_b) >> store

            graph = ctx.graph._graph

            # Find the parallel group
            parallel_groups = [n for n in graph.nodes() if n.startswith("ParallelGroup_")]
            assert len(parallel_groups) == 1, f"Expected 1 parallel group, got: {parallel_groups}"
            group_id = parallel_groups[0]

            # Verify: fetch -> ParallelGroup
            fetch_successors = list(graph.successors("fetch"))
            assert group_id in fetch_successors, \
                f"fetch should connect to ParallelGroup, got: {fetch_successors}"

            # Verify: ParallelGroup membership via API
            members = ctx.graph.get_parallel_group_members(group_id)
            assert set(members) == {"transform_a", "transform_b"}, \
                f"ParallelGroup members should be transform_a/transform_b, got: {members}"

            group_successors = list(graph.successors(group_id))
            assert group_successors == ["store"], \
                f"ParallelGroup should have only store as external successor, got: {group_successors}"

            # Verify: transform_a -> subtask_a
            transform_a_successors = list(graph.successors("transform_a"))
            assert "subtask_a" in transform_a_successors, \
                f"transform_a should connect to subtask_a, got: {transform_a_successors}"

            # Verify: ParallelGroup -> store
            assert "store" in group_successors, \
                f"ParallelGroup should connect to store, got: {group_successors}"

    def test_complex_chain_combination(self):
        """Test complex chain: (a >> b) >> (c >> d)."""
        with WorkflowContext("test") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")
            task_d = Task("task_d")

            # Create: (a >> b) >> (c >> d)
            chain1 = task_a >> task_b
            chain2 = task_c >> task_d
            result = chain1 >> chain2

            assert isinstance(result, SequentialTask)
            assert len(result.tasks) == 4
            assert result.leftmost == task_a
            assert result.rightmost == task_d

            graph = ctx.graph._graph

            # Verify edges: a -> b -> c -> d
            assert ("task_a", "task_b") in graph.edges()
            assert ("task_b", "task_c") in graph.edges()
            assert ("task_c", "task_d") in graph.edges()

    def test_parallel_group_with_chains(self):
        """Test: ((a >> b) | (c >> d)) >> e."""
        with WorkflowContext("test") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")
            task_d = Task("task_d")
            task_e = Task("task_e")

            # Create: ((a >> b) | (c >> d)) >> e
            _ =((task_a >> task_b) | (task_c >> task_d)) >> task_e

            graph = ctx.graph._graph

            # Find the parallel group
            parallel_groups = [n for n in graph.nodes() if n.startswith("ParallelGroup_")]
            assert len(parallel_groups) == 1
            group_id = parallel_groups[0]

            members = ctx.graph.get_parallel_group_members(group_id)
            assert set(members) == {"task_a", "task_c"}

            group_successors = list(graph.successors(group_id))
            assert group_successors == ["task_e"]

            # Verify: task_a -> task_b, task_c -> task_d
            assert ("task_a", "task_b") in graph.edges()
            assert ("task_c", "task_d") in graph.edges()

            # Verify: ParallelGroup -> task_e
            assert "task_e" in group_successors


class TestSequentialTaskExecution:
    """Test cases for SequentialTask execution."""

    def test_sequential_task_run(self, execution_context):
        """Test SequentialTask.run() executes leftmost task."""
        with WorkflowContext("test"):
            executed = []

            def task_func_a():
                executed.append("task_a")
                return "result_a"

            def task_func_b():
                executed.append("task_b")
                return "result_b"

            task_a = TaskWrapper("task_a", task_func_a)
            task_b = TaskWrapper("task_b", task_func_b)

            sequential = task_a >> task_b
            sequential.set_execution_context(execution_context)

            # run() should execute leftmost task
            result = sequential.run()

            # Only leftmost should be executed directly
            assert "task_a" in executed
            assert result == "result_a"

    def test_sequential_task_call(self, execution_context):
        """Test SequentialTask.__call__() executes leftmost task."""
        with WorkflowContext("test"):
            def task_func():
                return "result"

            task_a = TaskWrapper("task_a", task_func)
            task_b = Task("task_b")

            sequential = task_a >> task_b
            sequential.set_execution_context(execution_context)

            # __call__() should delegate to run()
            result = sequential()
            assert result == "result"


class TestSequentialTaskWithTaskWrapper:
    """Test cases for SequentialTask with TaskWrapper."""

    def test_sequential_task_with_task_wrapper(self):
        """Test SequentialTask works with TaskWrapper."""
        with WorkflowContext("test"):
            def task_func():
                return "result"

            task_wrapper = TaskWrapper("wrapper_task", task_func)
            task_b = Task("task_b")

            result = task_wrapper >> task_b

            assert isinstance(result, SequentialTask)
            assert result.leftmost == task_wrapper
            assert result.rightmost == task_b

    def test_parallel_with_task_wrapper_chain(self):
        """Test ParallelGroup creation with TaskWrapper chains."""
        with WorkflowContext("test"):
            def func_a():
                return "a"

            def func_b():
                return "b"

            wrapper_a = TaskWrapper("wrapper_a", func_a)
            wrapper_b = TaskWrapper("wrapper_b", func_b)
            task_c = Task("task_c")

            # (wrapper_a >> wrapper_b) | task_c
            result = (wrapper_a >> wrapper_b) | task_c

            assert isinstance(result, ParallelGroup)
            assert wrapper_a in result.tasks
            assert task_c in result.tasks


class TestSequentialTaskEdgeCases:
    """Test edge cases for SequentialTask."""

    def test_empty_tasks_list_raises_error(self):
        """Test SequentialTask raises error with empty tasks list."""
        with pytest.raises(ValueError, match="requires at least one task"):
            SequentialTask([])

    def test_single_task_chain(self):
        """Test SequentialTask with single task (edge case)."""
        with WorkflowContext("test"):
            task_a = Task("task_a")

            # Single task in tasks list
            sequential = SequentialTask([task_a])

            assert sequential.leftmost == task_a
            assert sequential.rightmost == task_a
            assert len(sequential.tasks) == 1

    def test_sequential_task_not_registered_in_graph(self):
        """Test SequentialTask is not registered as a graph node."""
        with WorkflowContext("test") as ctx:
            task_a = Task("task_a")
            task_b = Task("task_b")

            _sequential = task_a >> task_b

            graph = ctx.graph._graph

            # SequentialTask itself should not be a node
            sequential_nodes = [n for n in graph.nodes() if n.startswith("SequentialTask")]
            assert len(sequential_nodes) == 0, \
                f"SequentialTask should not be registered in graph, found: {sequential_nodes}"

            # But individual tasks should be registered
            assert "task_a" in graph.nodes()
            assert "task_b" in graph.nodes()

    def test_multiple_sequential_operations(self):
        """Test multiple sequential operations on same tasks."""
        with WorkflowContext("test"):
            task_a = Task("task_a")
            task_b = Task("task_b")
            task_c = Task("task_c")
            task_d = Task("task_d")

            # Create multiple chains
            chain1 = task_a >> task_b
            chain2 = task_c >> task_d

            # Combine chains
            result = chain1 >> chain2

            assert isinstance(result, SequentialTask)
            assert len(result.tasks) == 4
            assert [t.task_id for t in result.tasks] == ["task_a", "task_b", "task_c", "task_d"]


class TestSequentialTaskEngineExecution:
    """Test cases for SequentialTask execution with WorkflowEngine."""

    def test_simple_sequential_execution(self):
        """Test execution of task_a >> task_b >> task_c with WorkflowEngine."""
        execution_order = []

        with WorkflowContext("test") as ctx:
            @task
            def task_a():
                execution_order.append("task_a")
                return "result_a"

            @task
            def task_b():
                execution_order.append("task_b")
                return "result_b"

            @task
            def task_c():
                execution_order.append("task_c")
                return "result_c"

            # Build workflow: task_a >> task_b >> task_c
            task_a >> task_b >> task_c  # type: ignore

            # Create execution context
            exec_context = ExecutionContext.create(ctx.graph, max_steps=10)

            # Execute with WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(exec_context)

            # Verify execution order
            assert execution_order == ["task_a", "task_b", "task_c"]

            # Verify results stored in context
            assert exec_context.get_result("task_a") == "result_a"
            assert exec_context.get_result("task_b") == "result_b"
            assert exec_context.get_result("task_c") == "result_c"

    def test_sequential_with_task_wrappers(self):
        """Test execution of TaskWrapper chain with WorkflowEngine."""
        execution_order = []

        with WorkflowContext("test") as ctx:
            def func_a(x: int) -> int:
                execution_order.append("func_a")
                return x * 2

            def func_b(x: int) -> int:
                execution_order.append("func_b")
                return x + 10

            def func_c(x: int) -> int:
                execution_order.append("func_c")
                return x * 3

            # Create TaskWrappers
            task_a = TaskWrapper("task_a", lambda: func_a(5))
            task_b = TaskWrapper("task_b", lambda: func_b(10))
            task_c = TaskWrapper("task_c", lambda: func_c(20))

            # Build sequential workflow
            task_a >> task_b >> task_c  # type: ignore

            # Create execution context
            exec_context = ExecutionContext.create(ctx.graph, max_steps=10)

            # Execute with WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(exec_context)

            # Verify execution order
            assert execution_order == ["func_a", "func_b", "func_c"]

            # Verify results
            assert exec_context.get_result("task_a") == 10  # 5 * 2
            assert exec_context.get_result("task_b") == 20  # 10 + 10
            assert exec_context.get_result("task_c") == 60  # 20 * 3

    def test_sequential_with_branching(self):
        """Test execution of sequential tasks with parallel branching."""
        execution_order = []

        with WorkflowContext("test") as ctx:
            @task
            def start():
                execution_order.append("start")
                return "start"

            @task
            def branch_a():
                execution_order.append("branch_a")
                return "branch_a"

            @task
            def branch_b():
                execution_order.append("branch_b")
                return "branch_b"

            @task
            def merge():
                execution_order.append("merge")
                return "merge"

            # Build workflow: start >> (branch_a | branch_b) >> merge
            start >> (branch_a | branch_b) >> merge  # type: ignore

            # Create execution context
            exec_context = ExecutionContext.create(ctx.graph, start_node="start", max_steps=10)

            # Execute with WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(exec_context)

            # Verify start executed first
            assert execution_order[0] == "start"

            # Verify merge executed last
            assert execution_order[-1] == "merge"

            # Verify all tasks executed
            assert set(execution_order) == {"start", "branch_a", "branch_b", "merge"}

    def test_nested_parallel_execution(self):
        """Test execution of fetch >> ((transform_a >> subtask_a) | transform_b) >> store."""
        execution_order: list[str] = []

        with WorkflowContext("test") as ctx:
            @task
            def fetch():
                execution_order.append("fetch")
                return "fetched"

            @task
            def transform_a():
                execution_order.append("transform_a")
                return "transformed_a"

            @task
            def subtask_a():
                execution_order.append("subtask_a")
                return "subtask_a_done"

            @task
            def transform_b():
                execution_order.append("transform_b")
                return "transformed_b"

            @task
            def store():
                execution_order.append("store")
                return "stored"

            # Build nested parallel workflow
            fetch >> ((transform_a >> subtask_a) | transform_b) >> store  # type: ignore

            print(f"Graph: \n{ctx.graph}")

            exec_context = ExecutionContext.create(ctx.graph, start_node="fetch", max_steps=20)

            engine = WorkflowEngine()
            engine.execute(exec_context)

        print(f"Execution order: {execution_order}")

        assert len(execution_order) == 5
        assert execution_order[0] == "fetch"
        assert execution_order[-1] == "store"
        assert execution_order.count("transform_a") == 1
        assert execution_order.count("transform_b") == 1
        assert execution_order.count("subtask_a") == 1
        assert execution_order.count("store") == 1

        assert execution_order.index("transform_a") < execution_order.index("subtask_a")

        assert set(execution_order) == {"fetch", "transform_a", "transform_b", "subtask_a", "store"}

        for task_id in ("fetch", "transform_a", "transform_b", "subtask_a", "store"):
            result = exec_context.get_result(task_id)
            assert result is not None

    def test_nested_sequential_chains(self):
        """Test execution of nested sequential chains: (a >> b) >> (c >> d)."""
        execution_order = []

        with WorkflowContext("test") as ctx:
            @task
            def task_a():
                execution_order.append("a")
                return "a"

            @task
            def task_b():
                execution_order.append("b")
                return "b"

            @task
            def task_c():
                execution_order.append("c")
                return "c"

            @task
            def task_d():
                execution_order.append("d")
                return "d"

            # Build workflow: (a >> b) >> (c >> d)
            (task_a >> task_b) >> (task_c >> task_d)  # type: ignore

            # Create execution context
            exec_context = ExecutionContext.create(ctx.graph, max_steps=10)

            # Execute with WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(exec_context)

            # Verify execution order
            assert execution_order == ["a", "b", "c", "d"]

            # Verify all results
            for task_name in ["task_a", "task_b", "task_c", "task_d"]:
                assert exec_context.get_result(task_name) is not None

    def test_long_sequential_chain(self):
        """Test execution of long sequential chain to verify SequentialTask handling."""
        execution_order = []

        with WorkflowContext("test") as ctx:
            # Create 10 tasks in a chain
            tasks = []
            for i in range(10):
                @task(id=f"task_{i}")
                def task_func(idx=i):
                    execution_order.append(f"task_{idx}")
                    return f"result_{idx}"

                tasks.append(task_func)

            # Chain all tasks: task_0 >> task_1 >> task_2 >> ... >> task_9
            result = tasks[0]
            for t in tasks[1:]:
                result = result >> t  # type: ignore

            # Create execution context
            exec_context = ExecutionContext.create(ctx.graph, max_steps=20)

            # Execute with WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(exec_context)

            # Verify execution order
            expected_order = [f"task_{i}" for i in range(10)]
            assert execution_order == expected_order

            # Verify all results
            for i in range(10):
                assert exec_context.get_result(f"task_{i}") == f"result_{i}"

    def test_sequential_parallel_groups(self):
        """Test execution of sequential parallel groups: group1 >> group2 >> task5.

        This tests the pattern where multiple parallel groups are chained sequentially.
        Expected structure:
        - group1 -> [task1, task2]
        - group2 -> [task3, task4]
        - group1 >> group2 >> task5 creates sequential chain
        """
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            task4 = Task("task4")
            task5 = Task("task5")

            # Create sequential parallel groups
            group1 = (task1 | task2).set_group_name("group1")
            group2 = (task3 | task4).set_group_name("group2")
            group1 >> group2 >> task5  # type: ignore

            graph = ctx.graph._graph

            # Verify graph structure
            members_group1 = ctx.graph.get_parallel_group_members("group1")
            assert set(members_group1) == {"task1", "task2"}
            group1_successors = list(graph.successors("group1"))
            assert group1_successors == ["group2"], group1_successors

            # group2 membership and successor
            members_group2 = ctx.graph.get_parallel_group_members("group2")
            assert set(members_group2) == {"task3", "task4"}
            group2_successors = list(graph.successors("group2"))
            assert group2_successors == ["task5"], group2_successors

            # Verify no direct edges from group1 to task5
            assert "task5" not in group1_successors

    def test_multi_stage_workflow(self):
        """Test execution of multi-stage workflow: start >> group1 >> middle >> group2 >> end.

        This tests complex workflows with multiple parallel groups interspersed with
        single tasks.
        Expected structure:
        - start -> group1 -> [task1, task2] -> middle -> group2 -> [task3, task4] -> end
        """
        with WorkflowContext("test") as ctx:
            start = Task("start")
            task1 = Task("task1")
            task2 = Task("task2")
            middle = Task("middle")
            task3 = Task("task3")
            task4 = Task("task4")
            end = Task("end")

            # Multi-stage pattern
            group1 = (task1 | task2).set_group_name("group1")
            group2 = (task3 | task4).set_group_name("group2")
            start >> group1 >> middle >> group2 >> end  # type: ignore

            graph = ctx.graph._graph

            # Verify graph structure
            # start -> group1
            start_successors = list(graph.successors("start"))
            assert "group1" in start_successors

            # group1 membership and successor to middle
            members_group1 = ctx.graph.get_parallel_group_members("group1")
            assert set(members_group1) == {"task1", "task2"}
            group1_successors = list(graph.successors("group1"))
            assert group1_successors == ["middle"], group1_successors

            # middle -> group2
            middle_successors = list(graph.successors("middle"))
            assert "group2" in middle_successors

            # group2 membership and successor to end
            members_group2 = ctx.graph.get_parallel_group_members("group2")
            assert set(members_group2) == {"task3", "task4"}
            group2_successors = list(graph.successors("group2"))
            assert group2_successors == ["end"], group2_successors

            # Verify no shortcuts (e.g., start -> middle)
            assert "middle" not in start_successors
            assert "end" not in group1_successors
