"""Pytest-style tests for ParallelGroup functionality."""

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import ParallelGroup, SequentialTask, Task, TaskWrapper, parallel
from graflow.core.workflow import WorkflowContext


@pytest.fixture
def execution_context():
    """Create execution context for tests."""
    graph = TaskGraph()
    return ExecutionContext.create(graph, max_steps=10)


class TestParallelGroup:
    """Test cases for ParallelGroup class."""

    def test_parallel_group_creation(self):
        """Test ParallelGroup creation and basic properties."""
        with WorkflowContext("test_workflow"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])

            assert len(parallel_group.tasks) == 2
            assert task1 in parallel_group.tasks
            assert task2 in parallel_group.tasks
            assert parallel_group.task_id.startswith("ParallelGroup_")

    def test_parallel_group_or_operator(self):
        """Test | operator for creating parallel groups."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            parallel_group = task1 | task2 | task3

            assert isinstance(parallel_group, ParallelGroup)
            assert len(parallel_group.tasks) == 3

    def test_parallel_helper_creates_group(self):
        """parallel helper should mirror | semantics and register membership."""
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")

            parallel_group = parallel(task1, task2, task3)

            assert isinstance(parallel_group, ParallelGroup)
            assert set(parallel_group.tasks) == {task1, task2, task3}

            members = ctx.graph.get_parallel_group_members(parallel_group.task_id)
            assert set(members) == {"task1", "task2", "task3"}

    def test_parallel_group_run_with_default_executor(self, execution_context, mocker):
        """Test ParallelGroup.run() with default GroupExecutor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])
            parallel_group.set_execution_context(execution_context)

            mock_execute = mocker.patch.object(GroupExecutor, 'execute_parallel_group')
            parallel_group.run()

            mock_execute.assert_called_once()
            args, _kwargs = mock_execute.call_args
            group_id, tasks, context = args

            assert group_id == parallel_group.task_id
            assert len(tasks) == 2
            assert tasks[0].task_id == "task1"
            assert tasks[1].task_id == "task2"
            assert context == execution_context


    def test_parallel_group_with_task_wrapper(self, execution_context, mocker):
        """Test ParallelGroup with TaskWrapper that requires context injection."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            def test_func(ctx):
                return f"result_{ctx.task_id}"

            task_wrapper = TaskWrapper("wrapper_task", test_func, inject_context=True)
            parallel_group = ParallelGroup([task1, task_wrapper])
            parallel_group.set_execution_context(execution_context)

            mock_execute = mocker.patch.object(GroupExecutor, 'execute_parallel_group')
            parallel_group.run()

            mock_execute.assert_called_once()
            args, _kwargs = mock_execute.call_args
            _group_id, tasks, _context = args

            # Check that wrapper task has context function
            wrapper_task = tasks[1]
            assert wrapper_task.task_id == "wrapper_task"
            assert isinstance(wrapper_task, TaskWrapper)

    def test_parallel_group_dependency_operators(self):
        """Test >> and << operators with ParallelGroup."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])
            result_task = Task("result")

            # Test >> operator - now returns SequentialTask
            chained = parallel_group >> result_task
            assert isinstance(chained, SequentialTask)
            assert chained.leftmost == parallel_group
            assert chained.rightmost == result_task

            # Test << operator - now returns SequentialTask
            source_task = Task("source")
            chained = parallel_group << source_task
            assert isinstance(chained, SequentialTask)
            assert chained.leftmost == source_task
            assert chained.rightmost == parallel_group

    def test_parallel_group_rshift_creates_group_level_edge(self):
        """Test that >> creates edge from GROUP to successor, not from individual tasks.

        This is critical for preventing race conditions where multiple threads
        execute the successor task. Only the group should have the successor edge.
        """
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = task1 | task2
            successor = Task("successor")

            # Create dependency: (task1 | task2) >> successor
            parallel_group >> successor # type: ignore

            graph = ctx.graph._graph

            # Verify: Group has edge to successor
            group_successors = list(graph.successors(parallel_group.task_id))
            assert "successor" in group_successors, \
                f"Group should have successor edge, got: {group_successors}"

            # Verify: Individual tasks do NOT have edge to successor
            task1_successors = list(graph.successors("task1"))
            assert "successor" not in task1_successors, \
                f"task1 should NOT have successor edge, got: {task1_successors}"

            task2_successors = list(graph.successors("task2"))
            assert "successor" not in task2_successors, \
                f"task2 should NOT have successor edge, got: {task2_successors}"

    def test_parallel_group_lshift_creates_group_level_edge(self):
        """Test that << creates edge from predecessor to GROUP, not to individual tasks.

        This ensures the predecessor relationship is at the group level.
        """
        with WorkflowContext("test") as ctx:
            predecessor = Task("predecessor")
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = task1 | task2

            # Create dependency: predecessor >> (task1 | task2)
            parallel_group << predecessor # type: ignore

            graph = ctx.graph._graph

            # Verify: Predecessor has edge to group
            predecessor_successors = list(graph.successors("predecessor"))
            assert parallel_group.task_id in predecessor_successors, \
                f"Predecessor should have edge to group, got: {predecessor_successors}"

            # Verify: Predecessor does NOT have edges to individual tasks
            assert "task1" not in predecessor_successors, \
                f"Predecessor should NOT have edge to task1, got: {predecessor_successors}"
            assert "task2" not in predecessor_successors, \
                f"Predecessor should NOT have edge to task2, got: {predecessor_successors}"

    def test_parallel_diamond_pattern_graph_structure(self):
        """Test diamond pattern graph structure: fetch >> (A | B) >> store.

        This verifies the correct graph topology for preventing race conditions.
        Expected structure:
        - fetch -> ParallelGroup
        - ParallelGroup members tracked via TaskGraph.get_parallel_group_members
        - ParallelGroup -> store (NOT task_a/task_b -> store)
        """
        with WorkflowContext("test") as ctx:
            fetch = Task("fetch")
            task_a = Task("task_a")
            task_b = Task("task_b")
            store = Task("store")

            # Build: fetch >> (task_a | task_b) >> store
            fetch >> (task_a | task_b) >> store # type: ignore

            graph = ctx.graph._graph

            # Get the parallel group (created by |)
            parallel_groups = [n for n in graph.nodes() if n.startswith("ParallelGroup_")]
            assert len(parallel_groups) == 1, f"Should have exactly 1 parallel group, got: {parallel_groups}"
            group_id = parallel_groups[0]

            # Verify: fetch -> ParallelGroup
            fetch_successors = list(graph.successors("fetch"))
            assert group_id in fetch_successors, \
                f"fetch should connect to ParallelGroup, got: {fetch_successors}"

            # Verify: ParallelGroup membership via TaskGraph API
            members = ctx.graph.get_parallel_group_members(group_id)
            assert set(members) == {"task_a", "task_b"}, \
                f"ParallelGroup members should be task_a/task_b, got: {members}"

            # Verify: ParallelGroup only connects to external successors (store)
            group_successors = list(graph.successors(group_id))
            assert "store" in group_successors, \
                f"ParallelGroup should connect to store, got: {group_successors}"
            assert "task_a" not in group_successors, \
                f"ParallelGroup should not connect directly to task_a, got: {group_successors}"
            assert "task_b" not in group_successors, \
                f"ParallelGroup should not connect directly to task_b, got: {group_successors}"

            # Verify: task_a/task_b do NOT have successors (KEY FIX)
            task_a_successors = list(graph.successors("task_a"))
            assert len(task_a_successors) == 0, \
                f"task_a should have NO successors (prevent race), got: {task_a_successors}"

            task_b_successors = list(graph.successors("task_b"))
            assert len(task_b_successors) == 0, \
                f"task_b should have NO successors (prevent race), got: {task_b_successors}"

    def test_parallel_group_repr(self):
        """Test string representation of ParallelGroup."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])

            repr_str = repr(parallel_group)
            assert "ParallelGroup" in repr_str
            assert "task1" in repr_str
            assert "task2" in repr_str

    def test_parallel_group_merge_without_dependencies(self):
        """Test that groups without dependencies can be merged."""
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            task4 = Task("task4")

            # Create two groups without dependencies
            group1 = task1 | task2
            group2 = task3 | task4

            # Merge them - should extend group1
            result = group1 | group2

            # Should return group1 (merged)
            assert result == group1
            # All tasks should be in group1
            assert len(group1.tasks) == 4
            assert task1 in group1.tasks
            assert task2 in group1.tasks
            assert task3 in group1.tasks
            assert task4 in group1.tasks

            # group2 should be removed from graph
            graph = ctx.graph._graph
            group2_nodes = [n for n in graph.nodes() if n == group2.task_id]
            assert len(group2_nodes) == 0

    def test_parallel_group_merge_with_dependencies(self):
        """Test that groups with dependencies create a new ParallelGroup instead of merging."""
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            task4 = Task("task4")
            task5 = Task("task5")

            # Create group1 with a dependency
            group1 = (task1 | task2).set_group_name("group1")
            group1 >> task3  # type: ignore  # group1 now has task3 as a successor

            # Create group2 without dependency
            group2 = (task4 | task5).set_group_name("group2")

            # Merge them - should create a NEW ParallelGroup
            result = group1 | group2

            # Should create a new group
            assert result != group1
            assert result != group2
            assert isinstance(result, ParallelGroup)

            # New group should contain group1 and group2 as tasks
            assert len(result.tasks) == 2
            assert group1 in result.tasks
            assert group2 in result.tasks

            # Original groups should still exist
            graph = ctx.graph._graph
            assert "group1" in graph.nodes()
            assert "group2" in graph.nodes()

            # Verify group1 membership and external successor
            group1_members = ctx.graph.get_parallel_group_members("group1")
            assert set(group1_members) == {"task1", "task2"}
            group1_successors = list(graph.successors("group1"))
            assert group1_successors == ["task3"]

    def test_parallel_group_has_successors(self):
        """Test _has_successors() method correctly identifies dependencies."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")

            # Create group without successors
            group = task1 | task2
            assert not group._has_successors()

            # Add a successor
            group >> task3  # type: ignore

            # Now group should have successors
            assert group._has_successors()

    def test_complex_merge_pattern1(self):
        """Test complex merge: group1 >> task3 in separate line.

        Pattern:
            group1 = (task1 | task2).set_group_name("group1")
            group1 >> task3
            (group1 | (task4 | task5).set_group_name("group2")) >> task6

        Expected behavior:
            - group1 is ParallelGroup with dependency to task3
            - group1 | group2 creates new ParallelGroup_3
            - ParallelGroup_3 contains [group1, group2]
        """
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            task4 = Task("task4")
            task5 = Task("task5")
            task6 = Task("task6")

            # Create group1 and add dependency separately
            group1 = (task1 | task2).set_group_name("group1")
            group1 >> task3  # type: ignore

            # Create group2
            group2 = (task4 | task5).set_group_name("group2")

            # Merge and add task6
            (group1 | group2) >> task6  # type: ignore

            graph = ctx.graph._graph

            # Find ParallelGroup_3
            parallel_groups = [n for n in graph.nodes() if n.startswith("ParallelGroup_")]
            assert len(parallel_groups) == 1
            group3_id = parallel_groups[0]

            # Verify structure
            # group1 members + external successor task3
            group1_members = ctx.graph.get_parallel_group_members("group1")
            assert set(group1_members) == {"task1", "task2"}
            group1_successors = list(graph.successors("group1"))
            assert group1_successors == ["task3"]

            # group2 membership and successors
            group2_members = ctx.graph.get_parallel_group_members("group2")
            assert set(group2_members) == {"task4", "task5"}
            group2_successors = list(graph.successors("group2"))
            assert group2_successors == []

            # ParallelGroup_3 membership and successor
            group3_members = ctx.graph.get_parallel_group_members(group3_id)
            assert set(group3_members) == {"group1", "group2"}
            group3_successors = list(graph.successors(group3_id))
            assert group3_successors == ["task6"]

    def test_complex_merge_pattern2(self):
        """Test complex merge: group1 >> task3 in same expression.

        Pattern:
            group1 = (task1 | task2).set_group_name("group1") >> task3
            (group1 | (task4 | task5).set_group_name("group2")) >> task6

        Expected behavior:
            - group1 is SequentialTask([ParallelGroup("group1"), task3])
            - group1.leftmost is ParallelGroup("group1")
            - group1 | group2 uses leftmost (ParallelGroup("group1"))
            - Creates new ParallelGroup containing [group1.leftmost, group2]
        """
        with WorkflowContext("test") as ctx:
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            task4 = Task("task4")
            task5 = Task("task5")
            task6 = Task("task6")

            # Create group1 with dependency in same expression
            group1 = (task1 | task2).set_group_name("group1") >> task3

            # group1 should be SequentialTask
            assert isinstance(group1, SequentialTask)
            assert group1.leftmost.task_id == "group1"  # ParallelGroup
            assert group1.rightmost == task3

            # Create group2
            group2 = (task4 | task5).set_group_name("group2")

            # Merge and add task6
            # This should use group1.leftmost (ParallelGroup) in the merge
            # group1 | group2 returns group2 (with group1.leftmost added)
            merged = group1 | group2
            assert merged == group2
            assert group1.leftmost in merged.tasks

            merged >> task6  # type: ignore

            graph = ctx.graph._graph

            # Verify group1 (ParallelGroup) structure
            group1_members = ctx.graph.get_parallel_group_members("group1")
            assert set(group1_members) == {"task1", "task2"}
            group1_successors = list(graph.successors("group1"))
            assert group1_successors == ["task3"]

            # group2 should now include group1.leftmost alongside task4/task5
            group2_members = ctx.graph.get_parallel_group_members("group2")
            expected_members = {group1.leftmost.task_id, "task4", "task5"}
            assert set(group2_members) == expected_members
            group2_successors = list(graph.successors("group2"))
            assert group2_successors == ["task6"]


class TestParallelGroupIntegration:
    """Integration tests for ParallelGroup with real execution."""

    def test_direct_execution_integration(self, mocker):
        """Test ParallelGroup with direct execution backend."""
        results = []

        with WorkflowContext("test") as wf_ctx:
            def task_func_1():
                results.append("task1_executed")
                return "result1"

            def task_func_2():
                results.append("task2_executed")
                return "result2"

            task1 = TaskWrapper("task1", task_func_1)
            task2 = TaskWrapper("task2", task_func_2)

            parallel_group = ParallelGroup([task1, task2]).with_execution(backend=CoordinationBackend.DIRECT)

            # Use the workflow context's graph instead of creating a new one
            context = ExecutionContext.create(wf_ctx.graph, start_node="task1")

            parallel_group.set_execution_context(context)

            parallel_group.run()

            # Verify both tasks were executed (order may vary)
            assert set(results) == {"task1_executed", "task2_executed"}
            assert context.get_result("task1") == "result1"
            assert context.get_result("task2") == "result2"
