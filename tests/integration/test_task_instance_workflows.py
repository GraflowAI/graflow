"""Integration tests for multi-instance task workflows.

This module tests real-world scenarios where multiple task instances are created
from the same @task function and used in workflows.
"""


from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


class TestMultiInstanceWorkflows:
    """Test workflows with multiple task instances."""

    def test_multi_instance_sequential_workflow(self):
        """Test sequential workflow with multiple instances of same task."""
        results = []

        # Define task OUTSIDE workflow to avoid auto-registration
        @task
        def process_query(query: str) -> str:
            result = f"Processed: {query}"
            results.append(result)
            return result

        with workflow("multi_instance") as wf:
            # Create multiple instances with different queries
            ask_tokyo = process_query(task_id="tokyo", query="What's the weather in Tokyo?")
            ask_paris = process_query(task_id="paris", query="What's the weather in Paris?")
            ask_london = process_query(task_id="london", query="What's the weather in London?")

            # Chain them sequentially
            _ = ask_tokyo >> ask_paris >> ask_london

            # Execute workflow
            wf.execute(ask_tokyo.task_id)

        # Verify all tasks executed with their bound parameters
        assert len(results) == 3
        assert "Tokyo" in results[0]
        assert "Paris" in results[1]
        assert "London" in results[2]

    def test_multi_instance_parallel_workflow(self):
        """Test parallel workflow with multiple instances of same task."""
        results = []

        @task
        def fetch_data(source: str) -> dict:
            data = {"source": source, "data": f"data_from_{source}"}
            results.append(data)
            return data

        exec_context: ExecutionContext | None = None
        with workflow("parallel_fetch") as wf:
            # Create multiple instances for parallel execution
            fetch_api = fetch_data(task_id="api", source="api")
            fetch_db = fetch_data(task_id="db", source="database")
            fetch_cache = fetch_data(task_id="cache", source="cache")

            # Execute in parallel
            _parallel_group = fetch_api | fetch_db | fetch_cache

            _, exec_context = wf.execute(ret_context=True)

        # Verify all tasks executed
        assert len(results) == 3
        sources = [r["source"] for r in results]
        assert "api" in sources
        assert "database" in sources
        assert "cache" in sources

        # Verify results
        assert exec_context is not None
        execution_results = {
            task_id: exec_context.get_result(task_id)
            for task_id in ["api", "db", "cache"]
        }
        assert execution_results["api"]["source"] == "api"
        assert execution_results["db"]["source"] == "database"
        assert execution_results["cache"]["source"] == "cache"

    def test_multi_instance_diamond_workflow(self):
        """Test diamond-shaped workflow with multiple instances."""
        execution_log = []

        @task(inject_context=True)
        def transform(ctx, operation: str, value: int) -> int:
            if operation == "multiply":
                result = value * 2
            elif operation == "add":
                result = value + 10
            elif operation == "square":
                result = value ** 2
            else:
                result = value

            execution_log.append(f"{operation}: {value} -> {result}")
            channel = ctx.get_channel()
            channel.set(f"{operation}_result", result)
            return result

        @task(inject_context=True)
        def combine(ctx) -> dict:
            channel = ctx.get_channel()
            multiply_result = channel.get("multiply_result")
            add_result = channel.get("add_result")
            total = multiply_result + add_result
            execution_log.append(f"combine: {multiply_result} + {add_result} = {total}")
            return {"multiply": multiply_result, "add": add_result, "total": total}

        with workflow("diamond") as wf:
            # Create source task
            source = transform(task_id="source", operation="square", value=3)

            # Create parallel transforms
            multiply = transform(task_id="multiply", operation="multiply", value=9)
            add = transform(task_id="add", operation="add", value=9)

            # Create diamond pattern
            _ = source >> (multiply | add) >> combine

            _, exec_context = wf.execute(source.task_id, ret_context=True, initial_channel={"value": 9})

        # Verify execution using exec_context.get_result()
        assert exec_context.get_result("source") == 9  # 3^2
        assert exec_context.get_result("multiply") == 18  # 9 * 2
        assert exec_context.get_result("add") == 19  # 9 + 10
        assert exec_context.get_result("combine")["total"] == 37  # 18 + 19  # 18 + 19


class TestChannelOverride:
    """Test that bound parameters override channel values."""

    def test_bound_param_overrides_channel(self):
        """Test explicit bound parameter takes precedence over channel."""
        @task
        def process(value: int, multiplier: int) -> int:
            return value * multiplier

        with workflow("override_test") as wf:
            # Create task with explicit bound parameter
            task_instance = process(task_id="test", value=10)  # Override value, multiplier from channel

            _, exec_context = wf.execute(task_instance.task_id, ret_context=True, initial_channel={"value": 100, "multiplier": 5})

        # Should use bound value (10) and channel multiplier (5)
        assert exec_context.get_result("test") == 50

    def test_all_params_from_channel_when_not_bound(self):
        """Test that all params come from channel when none are bound."""
        @task
        def process(value: int, multiplier: int) -> int:
            return value * multiplier

        with workflow("channel_test") as wf:
            # Create task with no bound parameters
            task_instance = process(task_id="test")

            _, exec_context = wf.execute(task_instance.task_id, ret_context=True, initial_channel={"value": 7, "multiplier": 3})

        # Should use all channel values
        assert exec_context.get_result("test") == 21

    def test_mixed_bound_and_channel_params(self):
        """Test mixing bound parameters with channel resolution."""
        @task
        def calculate(a: int, b: int, c: int, d: int) -> int:
            return a + b + c + d

        with workflow("mixed_test") as wf:
            # Bind some parameters, leave others to channel
            task_instance = calculate(task_id="test", a=10, c=30)

            _, exec_context = wf.execute(task_instance.task_id, ret_context=True, initial_channel={"a": 1, "b": 2, "c": 3, "d": 4})

        # Should use bound (a=10, c=30) and channel (b=2, d=4)
        assert exec_context.get_result("test") == 46  # 10 + 2 + 30 + 4  # 10 + 2 + 30 + 4


class TestAutoGeneratedTaskIds:
    """Test workflows with auto-generated task IDs."""

    def test_workflow_with_auto_generated_ids(self):
        """Test that auto-generated task IDs work in workflows."""
        @task
        def process(value: int) -> int:
            return value * 2

        result = None
        with workflow("auto_id") as wf:
            # Create tasks with auto-generated IDs
            task1 = process(value=5)
            task2 = process(value=10)
            task3 = process(value=15)

            # Chain them
            _ = task1 >> task2 >> task3

            result = wf.execute()

        # Verify tasks executed (check by value pattern)
        assert result == 30  # Last task: 15 * 2

    def test_auto_generated_ids_are_unique(self):
        """Test that auto-generated IDs don't collide."""
        @task
        def process(value: int) -> int:
            return value

        with workflow("unique_ids"):
            # Create many instances with same parameters
            tasks = [process(value=i) for i in range(10)]

            # Get all task IDs
            task_ids = [t.task_id for t in tasks]

            # Verify all unique
            assert len(task_ids) == len(set(task_ids))
            assert all(tid.startswith("process_") for tid in task_ids)


class TestRealWorldScenario:
    """Test real-world scenario: data processing pipeline."""

    def test_data_processing_pipeline(self):
        """Test a realistic data processing pipeline with multiple instances."""

        @task(inject_context=True)
        def fetch_data(ctx, source: str) -> dict:
            """Simulate fetching data from different sources."""
            data = {
                "api": {"records": 100, "status": "success"},
                "database": {"records": 250, "status": "success"},
                "file": {"records": 50, "status": "success"}
            }
            result = data.get(source, {"records": 0, "status": "unknown"})
            channel = ctx.get_channel()
            channel.set(f"{source}_data", result)
            return result

        @task(inject_context=True)
        def validate_data(ctx, source: str) -> bool:
            """Validate fetched data."""
            channel = ctx.get_channel()
            data = channel.get(f"{source}_data")
            is_valid = data["status"] == "success" and data["records"] > 0
            channel.set(f"{source}_valid", is_valid)
            return is_valid

        @task(inject_context=True)
        def aggregate_results(ctx) -> dict:
            """Aggregate all validation results."""
            channel = ctx.get_channel()
            api_valid = channel.get("api_valid")
            db_valid = channel.get("database_valid")
            file_valid = channel.get("file_valid")

            api_data = channel.get("api_data")
            db_data = channel.get("database_data")
            file_data = channel.get("file_data")

            return {
                "all_valid": all([api_valid, db_valid, file_valid]),
                "total_records": sum([
                    api_data["records"],
                    db_data["records"],
                    file_data["records"]
                ]),
                "sources": ["api", "database", "file"]
            }

        with workflow("data_pipeline") as wf:
            # Create multiple fetch tasks
            fetch_api = fetch_data(task_id="fetch_api", source="api")
            fetch_db = fetch_data(task_id="fetch_db", source="database")
            fetch_file = fetch_data(task_id="fetch_file", source="file")

            # Create corresponding validation tasks
            validate_api = validate_data(task_id="validate_api", source="api")
            validate_db = validate_data(task_id="validate_db", source="database")
            validate_file = validate_data(task_id="validate_file", source="file")

            # Build pipeline: fetch in parallel, then validate in parallel, then aggregate
            fetch_all = fetch_api | fetch_db | fetch_file
            validate_all = validate_api | validate_db | validate_file
            _ = fetch_all >> validate_all >> aggregate_results

            _, exec_context = wf.execute(fetch_all.task_id, ret_context=True)

        # Verify results using exec_context.get_result()
        assert exec_context.get_result("fetch_api")["records"] == 100
        assert exec_context.get_result("fetch_db")["records"] == 250
        assert exec_context.get_result("fetch_file")["records"] == 50

        assert exec_context.get_result("validate_api") is True
        assert exec_context.get_result("validate_db") is True
        assert exec_context.get_result("validate_file") is True

        assert exec_context.get_result("aggregate_results")["all_valid"] is True
        assert exec_context.get_result("aggregate_results")["total_records"] == 400
        assert len(exec_context.get_result("aggregate_results")["sources"]) == 3
