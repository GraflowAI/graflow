"""
Comprehensive unit tests for Tasks and Workflows Guide examples.

This test file verifies all examples from docs/tutorial/tasks_and_workflows_guide.md,
ensuring that the documented behavior matches the actual implementation.

Note: LLM and HITL-specific tests are excluded and should be in separate test files.
"""

from typing import TypedDict

import pytest

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy, CriticalGroupPolicy
from graflow.core.task import chain, parallel
from graflow.core.workflow import workflow
from graflow.exceptions import GraflowWorkflowCanceledError

# =============================================================================
# Level 1: Your First Task
# =============================================================================


class TestLevel1FirstTask:
    """Tests for Level 1: Your First Task"""

    def test_basic_task_decorator(self):
        """Test basic @task decorator usage"""

        @task
        def hello():
            """A simple task."""
            return "Hello, Graflow!"

        result = hello.run()
        assert result == "Hello, Graflow!"

    def test_custom_task_id_via_instance(self):
        """Test custom task ID via task instance creation"""

        @task
        def hello():
            return "Hello!"

        # Create task instance with custom task_id
        greeting = hello(task_id="greeting_task")

        # Verify the task_id is set correctly
        assert greeting.task_id == "greeting_task"

    def test_task_run_with_parameters(self):
        """Test .run() method with parameters"""

        @task
        def calculate(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        result = calculate.run(x=5, y=3)
        assert result == 8

    def test_task_with_default_parameters(self):
        """Test task with default parameters"""

        @task
        def process_data(data: list[int], multiplier: int = 2) -> list[int]:
            """Process data with a multiplier."""
            return [x * multiplier for x in data]

        result1 = process_data.run(data=[1, 2, 3])
        assert result1 == [2, 4, 6]

        result2 = process_data.run(data=[1, 2, 3], multiplier=3)
        assert result2 == [3, 6, 9]


# =============================================================================
# Level 2: Your First Workflow
# =============================================================================


class TestLevel2FirstWorkflow:
    """Tests for Level 2: Your First Workflow"""

    def test_simple_workflow(self):
        """Test basic workflow execution"""
        execution_log = []

        with workflow("simple_pipeline") as wf:

            @task
            def start():
                execution_log.append("Starting!")
                return "start"

            @task
            def middle():
                execution_log.append("Middle!")
                return "middle"

            @task
            def end():
                execution_log.append("Ending!")
                return "end"

            start >> middle >> end  # type: ignore
            wf.execute()

        assert execution_log == ["Starting!", "Middle!", "Ending!"]


# =============================================================================
# Level 3: Task Composition
# =============================================================================


class TestLevel3TaskComposition:
    """Tests for Level 3: Task Composition"""

    def test_sequential_and_parallel_composition(self):
        """Test combining sequential and parallel operators"""
        execution_log = []

        with workflow("composition") as wf:

            @task
            def start():
                execution_log.append("Start")
                return "start"

            @task
            def parallel_a():
                execution_log.append("Parallel A")
                return "a"

            @task
            def parallel_b():
                execution_log.append("Parallel B")
                return "b"

            @task
            def end():
                execution_log.append("End")
                return "end"

            start >> (parallel_a | parallel_b) >> end  # type: ignore
            # Specify start node to avoid ambiguity with parallel group
            final_result = wf.execute(start_node="start")

        assert "Start" in execution_log
        assert "Parallel A" in execution_log
        assert "Parallel B" in execution_log
        assert "End" in execution_log
        assert execution_log.index("Start") < execution_log.index("End")
        assert final_result == "end"

    def test_chain_helper_function(self):
        """Test chain() helper for sequential tasks"""
        execution_log = []

        with workflow("chain_test") as wf:

            @task
            def task_a():
                execution_log.append("A")
                return "a"

            @task
            def task_b():
                execution_log.append("B")
                return "b"

            @task
            def task_c():
                execution_log.append("C")
                return "c"

            chain(task_a, task_b, task_c)  # type: ignore
            wf.execute()

        assert execution_log == ["A", "B", "C"]

    def test_parallel_helper_function(self):
        """Test parallel() helper for concurrent tasks"""
        execution_log = []

        with workflow("parallel_test") as wf:

            @task
            def task_a():
                execution_log.append("A")
                return "a"

            @task
            def task_b():
                execution_log.append("B")
                return "b"

            @task
            def task_c():
                execution_log.append("C")
                return "c"

            parallel(task_a, task_b, task_c)  # type: ignore
            wf.execute()

        # All tasks should execute
        assert set(execution_log) == {"A", "B", "C"}
        assert len(execution_log) == 3

    def test_parallel_group_with_custom_name(self):
        """Test setting custom group name"""

        with workflow("named_group") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                return "c"

            group = (task_a | task_b | task_c).set_group_name("my_parallel_tasks")  # type: ignore
            # Parallel group naming is an internal feature
            assert group.task_id == "my_parallel_tasks"
            wf.execute()

    def test_parallel_group_best_effort_policy(self):
        """Test best_effort execution policy"""

        with workflow("best_effort") as wf:

            @task
            def task_success():
                return "success"

            @task
            def task_fail():
                raise ValueError("Intentional failure")

            @task
            def task_another():
                return "another"

            # With best_effort, workflow should continue even if one task fails
            (task_success | task_fail | task_another).with_execution(  # type: ignore
                policy="best_effort"
            )

            # Should not raise exception
            wf.execute()

    def test_at_least_n_policy(self):
        """Test AtLeastNGroupPolicy"""

        with workflow("at_least_n") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                raise ValueError("Failed")

            @task
            def task_d():
                return "d"

            # Require at least 3 out of 4 to succeed
            (task_a | task_b | task_c | task_d).with_execution(  # type: ignore
                policy=AtLeastNGroupPolicy(min_success=3)
            )

            # Should succeed because 3 tasks succeed
            wf.execute()

    def test_critical_group_policy(self):
        """Test CriticalGroupPolicy"""

        with workflow("critical") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                raise ValueError("Failed")

            # task_a and task_b are critical, task_c can fail
            (task_a | task_b | task_c).with_execution(  # type: ignore
                policy=CriticalGroupPolicy(critical_task_ids=["task_a", "task_b"])
            )

            # Should succeed because critical tasks succeed
            wf.execute()


# =============================================================================
# Level 4: Passing Parameters
# =============================================================================


class TestLevel4PassingParameters:
    """Tests for Level 4: Passing Parameters"""

    def test_channel_communication(self):
        """Test basic channel communication between tasks"""

        with workflow("channel_communication") as wf:

            @task(inject_context=True)
            def producer(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                channel.set("user_id", "user_123")

            @task(inject_context=True)
            def consumer(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                user_id = channel.get("user_id")
                return user_id

            producer >> consumer  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result("consumer") == "user_123"

    def test_partial_parameter_binding(self):
        """Test binding some parameters while others come from channel"""

        with workflow("partial_binding") as wf:

            @task
            def calculate(base: int, multiplier: int, offset: int) -> int:
                result = base * multiplier + offset
                return result

            task_inst = calculate(task_id="calc", base=10)

            # Specify start node since task instance creates a node
            _, ctx = wf.execute(
                start_node="calc",
                ret_context=True,
                initial_channel={"multiplier": 3, "offset": 5}
            )

            result = ctx.get_result("calc")

        assert result == 35  # 10 * 3 + 5


# =============================================================================
# Level 5: Task Instances
# =============================================================================


class TestLevel5TaskInstances:
    """Tests for Level 5: Task Instances"""

    def test_task_instances_with_parameters(self):
        """Test creating multiple task instances with different parameters"""

        @task
        def fetch_weather(city: str) -> str:
            return f"Weather for {city}"

        with workflow("weather") as wf:
            tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
            paris = fetch_weather(task_id="paris", city="Paris")
            london = fetch_weather(task_id="london", city="London")

            tokyo >> paris >> london  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result("tokyo") == "Weather for Tokyo"
        assert ctx.get_result("paris") == "Weather for Paris"
        assert ctx.get_result("london") == "Weather for London"

    def test_auto_generated_task_ids(self):
        """Test auto-generated task IDs for instances"""

        @task
        def process(value: int) -> int:
            return value * 2

        task1 = process(value=10)
        task2 = process(value=20)
        task3 = process(value=30)

        # Task IDs should be auto-generated and unique
        assert task1.task_id != task2.task_id
        assert task2.task_id != task3.task_id
        assert task1.task_id.startswith("process_")
        assert task2.task_id.startswith("process_")
        assert task3.task_id.startswith("process_")

        with workflow("auto_ids") as wf:
            task1 >> task2 >> task3  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result(task1.task_id) == 20
        assert ctx.get_result(task2.task_id) == 40
        assert ctx.get_result(task3.task_id) == 60

    def test_unique_task_ids_required(self):
        """Test that unique task IDs work correctly"""

        @task
        def fetch_weather(city: str) -> str:
            return f"Weather for {city}"

        # Good: Unique task_ids
        tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
        paris = fetch_weather(task_id="paris", city="Paris")

        assert tokyo.task_id == "tokyo"
        assert paris.task_id == "paris"


# =============================================================================
# Level 6: Channels and Context
# =============================================================================


class TestLevel6ChannelsAndContext:
    """Tests for Level 6: Channels and Context"""

    def test_basic_channel_operations(self):
        """Test basic channel set/get operations"""

        with workflow("basic_channel") as wf:

            @task(inject_context=True)
            def producer(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                channel.set("user_id", "user_123")
                channel.set("score", 95.5)
                channel.set("active", True)
                channel.set("user_profile", {"name": "Alice", "age": 30})

            @task(inject_context=True)
            def consumer(ctx: TaskExecutionContext) -> dict:
                channel = ctx.get_channel()
                return {
                    "user_id": channel.get("user_id"),
                    "score": channel.get("score"),
                    "active": channel.get("active"),
                    "profile": channel.get("user_profile"),
                    "missing": channel.get("missing", default="default_value"),
                }

            producer >> consumer  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        result = ctx.get_result("consumer")
        assert result["user_id"] == "user_123"
        assert result["score"] == 95.5
        assert result["active"] is True
        assert result["profile"] == {"name": "Alice", "age": 30}
        assert result["missing"] == "default_value"

    def test_channel_list_operations(self):
        """Test channel append/prepend operations"""

        with workflow("list_operations") as wf:

            @task(inject_context=True)
            def collect_logs(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                channel.append("logs", "Log entry 1")
                channel.append("logs", "Log entry 2")
                channel.append("logs", "Log entry 3")
                return channel.get("logs")

            @task(inject_context=True)
            def use_stack(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                channel.prepend("stack", "First")
                channel.prepend("stack", "Second")
                channel.prepend("stack", "Third")
                return channel.get("stack")

            logs_task = collect_logs(task_id="logs")
            stack_task = use_stack(task_id="stack")

            grp = parallel(logs_task, stack_task)  # type: ignore

            _, ctx = wf.execute(start_node=grp.task_id, ret_context=True)

        logs = ctx.get_result("logs")
        stack = ctx.get_result("stack")

        assert logs == ["Log entry 1", "Log entry 2", "Log entry 3"]
        assert stack == ["Third", "Second", "First"]

    def test_channel_ttl(self):
        """Test TTL (time-to-live) for channel values"""
        import time

        with workflow("ttl_test") as wf:

            @task(inject_context=True)
            def cache_data(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                # Set with very short TTL
                channel.set("temp_value", "expires_soon", ttl=1)
                channel.set("permanent", "stays")

            @task(inject_context=True)
            def check_cache(ctx: TaskExecutionContext) -> dict:
                channel = ctx.get_channel()
                time.sleep(2)  # Wait for TTL to expire
                temp = channel.get("temp_value", default="expired")
                permanent = channel.get("permanent")
                return {"temp": temp, "permanent": permanent}

            cache_data >> check_cache  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        result = ctx.get_result("check_cache")
        assert result["temp"] == "expired"
        assert result["permanent"] == "stays"

    def test_typed_channel(self):
        """Test type-safe channel with TypedDict"""

        class UserProfile(TypedDict):
            user_id: str
            name: str
            email: str
            age: int
            premium: bool

        with workflow("typed_channel") as wf:

            @task(inject_context=True)
            def collect_user_data(ctx: TaskExecutionContext):
                typed_channel = ctx.get_typed_channel(UserProfile)
                user_profile: UserProfile = {
                    "user_id": "user_123",
                    "name": "Alice",
                    "email": "alice@example.com",
                    "age": 30,
                    "premium": True,
                }
                typed_channel.set("current_user", user_profile)

            @task(inject_context=True)
            def process_user_data(ctx: TaskExecutionContext) -> dict:
                typed_channel = ctx.get_typed_channel(UserProfile)
                user: UserProfile = typed_channel.get("current_user")  # type: ignore
                return {"name": user["name"], "email": user["email"]}

            collect_user_data >> process_user_data  # type: ignore
            _, ctx = wf.execute(ret_context=True)

        result = ctx.get_result("process_user_data")
        assert result["name"] == "Alice"
        assert result["email"] == "alice@example.com"

    def test_context_injection(self):
        """Test context injection for accessing channels and metadata"""

        with workflow("context_injection") as wf:

            @task(inject_context=True)
            def my_task(ctx: TaskExecutionContext, value: int):
                # Access channel
                channel = ctx.get_channel()
                channel.set("result", value * 2)

                # Access session info
                assert ctx.session_id is not None

                return value * 2

            result = wf.execute(initial_channel={"value": 5})

        assert result == 10

    def test_parameter_priority(self):
        """Test parameter resolution priority: Bound > Channel"""

        with workflow("priority") as wf:

            @task
            def calculate(value: int, multiplier: int) -> int:
                return value * multiplier

            # Bind value=10, multiplier from channel
            task_inst = calculate(task_id="calc", value=10)

            result = wf.execute(start_node="calc", initial_channel={"value": 100, "multiplier": 5})

        # Bound value (10) beats channel value (100)
        assert result == 50


# =============================================================================
# Level 7: Execution Patterns
# =============================================================================


class TestLevel7ExecutionPatterns:
    """Tests for Level 7: Execution Patterns"""

    def test_get_final_result(self):
        """Test getting final result from workflow"""

        with workflow("simple") as wf:

            @task
            def compute():
                return 42

            result = wf.execute()

        assert result == 42

    def test_get_all_results(self):
        """Test getting all task results using ret_context"""

        with workflow("all_results") as wf:

            @task
            def task_a():
                return "A"

            @task
            def task_b():
                return "B"

            task_a >> task_b  # type: ignore

            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result("task_a") == "A"
        assert ctx.get_result("task_b") == "B"

    def test_auto_start_node_detection(self):
        """Test automatic start node detection"""

        with workflow("auto_start") as wf:

            @task
            def step1():
                return "Step 1"

            @task
            def step2():
                return "Step 2"

            step1 >> step2  # type: ignore

            # Auto-detects step1 (node with no predecessors)
            result = wf.execute()

        assert result == "Step 2"

    def test_manual_start_node(self):
        """Test manually specifying start node"""
        execution_log = []

        with workflow("manual_start") as wf:

            @task
            def step1():
                execution_log.append("Step 1")
                return "step1"

            @task
            def step2():
                execution_log.append("Step 2")
                return "step2"

            @task
            def step3():
                execution_log.append("Step 3")
                return "step3"

            step1 >> step2 >> step3  # type: ignore

            # Start from step2 (skip step1)
            result = wf.execute(start_node="step2")

        assert "Step 1" not in execution_log
        assert "Step 2" in execution_log
        assert "Step 3" in execution_log
        assert result == "step3"

    def test_result_storage_format(self):
        """Test that results are stored and accessible via get_result"""

        with workflow("result_format") as wf:

            @task
            def calculate():
                return 42

            task1 = calculate(task_id="calc1")
            task2 = calculate(task_id="calc2")

            grp = parallel(task1, task2)  # type: ignore

            _, ctx = wf.execute(start_node=grp.task_id, ret_context=True)

        # Results should be accessible via get_result
        assert ctx.get_result("calc1") == 42
        assert ctx.get_result("calc2") == 42


# =============================================================================
# Level 8: Complex Workflows
# =============================================================================


class TestLevel8ComplexWorkflows:
    """Tests for Level 8: Complex Workflows"""

    def test_diamond_pattern(self):
        """Test diamond pattern: split -> parallel -> merge"""

        with workflow("diamond") as wf:

            @task(inject_context=True)
            def source(ctx: TaskExecutionContext, value: int) -> int:
                ctx.get_channel().set("value", value)
                return value

            @task(inject_context=True)
            def double(ctx: TaskExecutionContext) -> int:
                value = ctx.get_channel().get("value")
                result = value * 2
                ctx.get_channel().set("doubled", result)
                return result

            @task(inject_context=True)
            def triple(ctx: TaskExecutionContext) -> int:
                value = ctx.get_channel().get("value")
                result = value * 3
                ctx.get_channel().set("tripled", result)
                return result

            @task(inject_context=True)
            def combine(ctx: TaskExecutionContext) -> int:
                doubled = ctx.get_channel().get("doubled")
                tripled = ctx.get_channel().get("tripled")
                return doubled + tripled

            src = source(task_id="src", value=5)
            src >> (double | triple) >> combine  # type: ignore

            result = wf.execute(start_node="src")

        assert result == 25  # 5*2 + 5*3

    def test_multi_instance_pipeline(self):
        """Test processing multiple items in parallel"""

        with workflow("multi_pipeline") as wf:

            @task
            def fetch(source: str) -> dict:
                return {"source": source, "data": f"data_{source}"}

            fetch_a = fetch(task_id="fetch_a", source="api")
            fetch_b = fetch(task_id="fetch_b", source="db")
            fetch_c = fetch(task_id="fetch_c", source="file")

            grp = parallel(fetch_a, fetch_b, fetch_c)  # type: ignore

            _, ctx = wf.execute(start_node=grp.task_id, ret_context=True)

        assert ctx.get_result("fetch_a") == {"source": "api", "data": "data_api"}
        assert ctx.get_result("fetch_b") == {"source": "db", "data": "data_db"}
        assert ctx.get_result("fetch_c") == {"source": "file", "data": "data_file"}


# =============================================================================
# Level 9: Dynamic Task Generation
# =============================================================================


class TestLevel9DynamicTaskGeneration:
    """Tests for Level 9: Dynamic Task Generation"""

    def test_next_iteration(self):
        """Test self-looping with next_iteration for convergence"""

        def train_step(accuracy):
            """Simulate training step"""
            return min(accuracy + 0.15, 0.98)

        with workflow("optimization") as wf:

            @task(inject_context=True)
            def optimize(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                iteration = channel.get("iteration", default=0)
                accuracy = channel.get("accuracy", default=0.5)

                # Training step
                new_accuracy = train_step(accuracy)

                if new_accuracy >= 0.95:
                    # Converged!
                    channel.set("final_accuracy", new_accuracy)
                    return new_accuracy
                else:
                    # Continue iterating
                    channel.set("iteration", iteration + 1)
                    channel.set("accuracy", new_accuracy)
                    ctx.next_iteration()
                    return None

            _, ctx = wf.execute(ret_context=True)

        final_accuracy = ctx.get_channel().get("final_accuracy")
        assert final_accuracy >= 0.95

    def test_cancel_workflow(self):
        """Test abnormal workflow cancellation"""
        execution_log = []

        with workflow("validation") as wf:

            @task(inject_context=True)
            def validate_data(ctx: TaskExecutionContext, data: dict):
                if not data.get("valid"):
                    execution_log.append("Validation failed")
                    ctx.cancel_workflow("Data validation failed")
                return data

            @task
            def process_data(data: dict):
                execution_log.append("Processing data...")
                return data

            validate = validate_data(task_id="validate", data={"valid": False})
            validate >> process_data  # type: ignore

            with pytest.raises(GraflowWorkflowCanceledError):
                wf.execute(start_node="validate")

        assert "Validation failed" in execution_log
        assert "Processing data..." not in execution_log


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests combining multiple concepts"""

    def test_complex_diamond_with_channels(self):
        """Test diamond pattern with channels"""

        with workflow("typed_diamond") as wf:

            @task(inject_context=True)
            def source(ctx: TaskExecutionContext):
                ctx.get_channel().set("input_value", 5)
                return 5

            @task(inject_context=True)
            def double(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                value = channel.get("input_value", default=5)
                result = value * 2
                channel.set("doubled", result)
                return result

            @task(inject_context=True)
            def triple(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                value = channel.get("input_value", default=5)
                result = value * 3
                channel.set("tripled", result)
                return result

            @task(inject_context=True)
            def combine(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                doubled = channel.get("doubled")
                tripled = channel.get("tripled")
                combined = doubled + tripled
                channel.set("combined", combined)
                return combined

            source >> (double | triple) >> combine  # type: ignore
            result = wf.execute(start_node="source")

        assert result == 25

    def test_multi_stage_pipeline(self):
        """Test multi-stage ETL pipeline"""

        with workflow("etl_pipeline") as wf:

            @task(inject_context=True)
            def extract(ctx: TaskExecutionContext, source: str) -> dict:
                result = {"source": source, "data": [1, 2, 3]}
                ctx.get_channel().set(f"extracted_{source}", result)
                return result

            @task(inject_context=True)
            def transform(ctx: TaskExecutionContext, source: str) -> dict:
                data = ctx.get_channel().get(f"extracted_{source}")
                data["transformed"] = [x * 2 for x in data["data"]]
                ctx.get_channel().set(f"transformed_{source}", data)
                return data

            @task(inject_context=True)
            def load(ctx: TaskExecutionContext, source: str):
                data = ctx.get_channel().get(f"transformed_{source}")
                ctx.get_channel().append("loaded_results", data)
                return data

            # Build a linear pipeline for one source
            extract(task_id="extract_api", source="api") >> transform(  # type: ignore
                task_id="transform_api", source="api"
            ) >> load(task_id="load_api", source="api")  # type: ignore

            _, ctx = wf.execute(start_node="extract_api", ret_context=True)

        # Verify all stages completed
        extract_result = ctx.get_result("extract_api")
        assert extract_result["source"] == "api"

        transform_result = ctx.get_result("transform_api")
        assert transform_result["transformed"] == [2, 4, 6]

        load_result = ctx.get_result("load_api")
        assert "transformed" in load_result
