"""Tests for automatic keyword argument resolution from channel."""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


class TestKeywordArgsResolution:
    """Test automatic keyword argument resolution from channel."""

    def test_basic_resolution(self):
        """Test basic keyword argument resolution from channel."""
        results = []

        with workflow("test_basic") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel data."""
                channel = context.get_channel()
                channel.set("name", "test")
                channel.set("value", 42)

            @task
            def process_data(name: str, value: int) -> None:
                """Process data with keyword arguments."""
                # Validate arguments are resolved correctly
                assert isinstance(name, str), f"Expected name to be str, got {type(name)}"
                assert isinstance(value, int), f"Expected value to be int, got {type(value)}"
                assert name == "test", f"Expected name='test', got {name!r}"
                assert value == 42, f"Expected value=42, got {value}"
                results.append({"name": name, "value": value})

            _ = setup >> process_data
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {"name": "test", "value": 42}

    def test_resolution_with_defaults(self):
        """Test resolution with default values."""
        results = []

        with workflow("test_defaults") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel data."""
                channel = context.get_channel()
                # Only set required parameter
                channel.set("required", "test_value")

            @task
            def process(required: str, optional: int = 10) -> None:
                """Process with optional parameter."""
                assert isinstance(required, str), f"Expected required to be str, got {type(required)}"
                assert isinstance(optional, int), f"Expected optional to be int, got {type(optional)}"
                assert required == "test_value"
                assert optional == 10
                results.append({"required": required, "optional": optional})

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {"required": "test_value", "optional": 10}

    def test_resolution_with_inject_context(self):
        """Test resolution combined with inject_context."""
        results = []

        with workflow("test_inject") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel data."""
                channel = context.get_channel()
                channel.set("data", "hello")
                channel.set("multiplier", 3)

            @task(inject_context=True)
            def process(context: TaskExecutionContext, data: str, multiplier: int = 2) -> None:
                """Process with both context and keyword args."""
                assert isinstance(context, TaskExecutionContext)
                assert isinstance(data, str), f"Expected data to be str, got {type(data)}"
                assert isinstance(multiplier, int), f"Expected multiplier to be int, got {type(multiplier)}"
                assert data == "hello", f"Expected data='hello', got {data!r}"
                assert multiplier == 3, f"Expected multiplier=3, got {multiplier}"
                results.append(
                    {
                        "session_id": context.session_id,
                        "data": data,
                        "multiplier": multiplier,
                    }
                )

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0]["data"] == "hello"
        assert results[0]["multiplier"] == 3
        assert results[0]["session_id"] is not None

    def test_disable_resolution(self):
        """Test disabling keyword argument resolution."""
        results = []

        with workflow("test_disable") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel but resolution is disabled."""
                channel = context.get_channel()
                # This value should NOT be used
                channel.set("value", "from_channel")

            @task(resolve_keyword_args=False)
            def process(value: str = "default") -> None:
                """Process without resolution."""
                assert isinstance(value, str), f"Expected value to be str, got {type(value)}"
                results.append(value)

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        # Should use default, not channel value
        assert results[0] == "default"

    def test_keyword_only_parameters(self):
        """Test resolution with keyword-only parameters."""
        results = []

        with workflow("test_keyword_only") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel data."""
                channel = context.get_channel()
                channel.set("name", "test")
                channel.set("value", 20)

            @task
            def process(*, name: str, value: int = 10) -> None:
                """Process with keyword-only parameters."""
                assert isinstance(name, str), f"Expected name to be str, got {type(name)}"
                assert isinstance(value, int), f"Expected value to be int, got {type(value)}"
                assert name == "test", f"Expected name='test', got {name!r}"
                assert value == 20, f"Expected value=20, got {value}"
                results.append({"name": name, "value": value})

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {"name": "test", "value": 20}

    def test_multiple_parameters(self):
        """Test resolution with multiple parameters of different types."""
        results = []

        with workflow("test_multiple") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup channel data."""
                channel = context.get_channel()
                channel.set("text", "hello")
                channel.set("count", 5)
                channel.set("flag", True)
                channel.set("ratio", 3.14)

            @task
            def process(text: str, count: int, flag: bool, ratio: float) -> None:
                """Process with multiple parameter types."""
                assert isinstance(text, str), f"Expected text to be str, got {type(text)}"
                assert isinstance(count, int), f"Expected count to be int, got {type(count)}"
                assert isinstance(flag, bool), f"Expected flag to be bool, got {type(flag)}"
                assert isinstance(ratio, float), f"Expected ratio to be float, got {type(ratio)}"
                assert text == "hello", f"Expected text='hello', got {text!r}"
                assert count == 5, f"Expected count=5, got {count}"
                assert flag is True, f"Expected flag=True, got {flag}"
                assert ratio == 3.14, f"Expected ratio=3.14, got {ratio}"
                results.append(
                    {
                        "text": text,
                        "count": count,
                        "flag": flag,
                        "ratio": ratio,
                    }
                )

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {
            "text": "hello",
            "count": 5,
            "flag": True,
            "ratio": 3.14,
        }

    def test_partial_resolution(self):
        """Test when only some parameters are in channel."""
        results = []

        with workflow("test_partial") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup partial data."""
                channel = context.get_channel()
                channel.set("name", "Alice")
                # Don't set 'age' - should use default

            @task
            def process(name: str, age: int = 18) -> None:
                """Process with partial data."""
                assert isinstance(name, str), f"Expected name to be str, got {type(name)}"
                assert isinstance(age, int), f"Expected age to be int, got {type(age)}"
                assert name == "Alice", f"Expected name='Alice', got {name!r}"
                assert age == 18, f"Expected age=18, got {age}"
                results.append({"name": name, "age": age})

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {"name": "Alice", "age": 18}

    def test_context_and_multiple_kwargs(self):
        """Test inject_context with multiple keyword argument resolution."""
        results = []

        with workflow("test_context_kwargs") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup multiple values in channel."""
                channel = context.get_channel()
                channel.set("name", "Alice")
                channel.set("age", 25)
                channel.set("city", "Tokyo")

            @task(inject_context=True)
            def process(context: TaskExecutionContext, name: str, age: int, city: str = "Unknown") -> None:
                """Process with context and multiple keyword args."""
                assert isinstance(context, TaskExecutionContext)
                assert isinstance(name, str), f"Expected name to be str, got {type(name)}"
                assert isinstance(age, int), f"Expected age to be int, got {type(age)}"
                assert isinstance(city, str), f"Expected city to be str, got {type(city)}"
                assert name == "Alice", f"Expected name='Alice', got {name!r}"
                assert age == 25, f"Expected age=25, got {age}"
                assert city == "Tokyo", f"Expected city='Tokyo', got {city!r}"
                results.append(
                    {
                        "session": context.session_id,
                        "name": name,
                        "age": age,
                        "city": city,
                    }
                )

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0]["session"] is not None
        assert results[0]["name"] == "Alice"
        assert results[0]["age"] == 25
        assert results[0]["city"] == "Tokyo"


class TestKeywordArgsResolutionEdgeCases:
    """Test edge cases for keyword argument resolution."""

    def test_empty_channel(self):
        """Test with empty channel and default values."""
        results = []

        with workflow("test_empty") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Don't set anything in channel."""
                # Channel is empty
                pass

            @task
            def process(value: str = "default") -> None:
                """Process with default."""
                assert isinstance(value, str), f"Expected value to be str, got {type(value)}"
                assert value == "default", f"Expected value='default', got {value!r}"
                results.append(value)

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == "default"

    def test_none_value_in_channel(self):
        """Test handling of None value in channel."""
        results = []

        with workflow("test_none") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup with None value."""
                channel = context.get_channel()
                # Explicitly set to None
                channel.set("value", None)

            @task
            def process(value: str = "default") -> None:
                """Process."""
                assert value is None
                results.append(value)

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        # None in channel should be passed as None
        assert results[0] is None

    def test_parameter_order_independence(self):
        """Test that resolution works regardless of parameter definition order."""
        results = []

        with workflow("test_order") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup in different order."""
                channel = context.get_channel()
                channel.set("a", 1)
                channel.set("m", 2.5)
                channel.set("z", "test")

            @task
            def process(z: str, a: int, m: float) -> None:
                """Process with non-alphabetical parameters."""
                assert isinstance(z, str), f"Expected z to be str, got {type(z)}"
                assert isinstance(a, int), f"Expected a to be int, got {type(a)}"
                assert isinstance(m, float), f"Expected m to be float, got {type(m)}"
                assert z == "test", f"Expected z='test', got {z!r}"
                assert a == 1, f"Expected a=1, got {a}"
                assert m == 2.5, f"Expected m=2.5, got {m}"
                results.append({"z": z, "a": a, "m": m})

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        assert results[0] == {"z": "test", "a": 1, "m": 2.5}

    def test_no_type_conversion(self):
        """Test that no automatic type conversion happens."""
        results = []

        with workflow("test_type") as wf:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext) -> None:
                """Setup with string value for int parameter."""
                channel = context.get_channel()
                # Set string value - will be passed as-is
                channel.set("value", "42")

            @task
            def process(value) -> None:
                """Process - accepts any type."""
                assert value == "42"
                assert isinstance(value, str), f"Expected value to be str, got {type(value)}"
                results.append(value)

            _ = setup >> process
            wf.execute(setup.task_id)

        assert len(results) == 1
        # No conversion - string is passed as-is
        assert results[0] == "42"
        assert isinstance(results[0], str)
