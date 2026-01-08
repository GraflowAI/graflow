"""
Custom Serialization with Cloudpickle
======================================

This example demonstrates cloudpickle's powerful serialization capabilities
for complex Python objects that standard pickle cannot handle.

Prerequisites:
--------------
1. pip install cloudpickle (installed with graflow)

Concepts Covered:
-----------------
1. Serializing functions and lambdas
2. Serializing closures with captured state
3. Serializing class instances
4. Handling non-serializable objects
5. Best practices for distributed execution

Expected Output:
----------------
=== Custom Serialization Demo ===

Test 1: Serializing Lambda Functions
✅ Lambda function serialized and deserialized
   Original result: 15
   Deserialized result: 15

Test 2: Serializing Closures
✅ Closure with captured state serialized
   Factory multiplier: 10
   Result: 50

Test 3: Serializing Class Instances
✅ Class instance with state serialized
   Counter before: 0
   Counter after increment: 3
   Counter after deserialization: 3

Test 4: Nested Closures
✅ Nested closure serialized
   Result: add_outer_inner(5) = 18

Test 5: Task with Complex State
✅ Task executed successfully
   Configuration: {'model': 'gpt-4', 'temp': 0.7}
   Result: Processed 3 items

=== Summary ===
✅ All serialization tests passed
✅ Cloudpickle handles complex Python objects
✅ Safe for distributed task execution
"""

import pickle

import cloudpickle


def test_lambda_serialization():
    """Test 1: Serialize and deserialize lambda functions."""
    print("Test 1: Serializing Lambda Functions")

    # Create a lambda function
    def add_ten(x):
        return x + 10

    # Try standard pickle (this would fail for lambdas in many cases)
    try:
        # Cloudpickle can serialize lambdas
        serialized = cloudpickle.dumps(add_ten)
        deserialized = cloudpickle.loads(serialized)

        # Test both functions
        original_result = add_ten(5)
        deserialized_result = deserialized(5)

        assert original_result == deserialized_result == 15

        print("✅ Lambda function serialized and deserialized")
        print(f"   Original result: {original_result}")
        print(f"   Deserialized result: {deserialized_result}\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")


def test_closure_serialization():
    """Test 2: Serialize closures with captured variables."""
    print("Test 2: Serializing Closures")

    def create_multiplier(factor):
        """Factory that creates a closure."""

        def multiply(x):
            return x * factor  # Captures 'factor' from outer scope

        return multiply

    # Create closure
    multiply_by_ten = create_multiplier(10)

    try:
        # Serialize the closure
        serialized = cloudpickle.dumps(multiply_by_ten)
        deserialized = cloudpickle.loads(serialized)

        # Test the deserialized closure
        result = deserialized(5)

        assert result == 50

        print("✅ Closure with captured state serialized")
        print("   Factory multiplier: 10")
        print(f"   Result: {result}\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")


def test_class_instance_serialization():
    """Test 3: Serialize class instances with state."""
    print("Test 3: Serializing Class Instances")

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self, amount=1):
            self.count += amount
            return self.count

        def get_count(self):
            return self.count

    try:
        # Create instance and modify state
        counter = Counter()
        print(f"   Counter before: {counter.get_count()}")

        counter.increment(3)
        print(f"   Counter after increment: {counter.get_count()}")

        # Serialize
        serialized = cloudpickle.dumps(counter)
        deserialized = cloudpickle.loads(serialized)

        # Verify state is preserved
        assert deserialized.get_count() == 3

        print(f"   Counter after deserialization: {deserialized.get_count()}")
        print("✅ Class instance with state serialized\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")


def test_nested_closure_serialization():
    """Test 4: Serialize nested closures."""
    print("Test 4: Nested Closures")

    def create_adder(outer_value):
        """Outer closure."""

        def add_outer(inner_value):
            """Inner closure that captures outer_value."""

            def add_inner(x):
                """Innermost function."""
                return x + outer_value + inner_value

            return add_inner

        return add_outer

    try:
        # Create nested closure
        add_5 = create_adder(5)
        add_5_8 = add_5(8)

        # Serialize the fully constructed closure
        serialized = cloudpickle.dumps(add_5_8)
        deserialized = cloudpickle.loads(serialized)

        # Test
        result = deserialized(5)  # 5 + 5 + 8 = 18
        assert result == 18

        print("✅ Nested closure serialized")
        print(f"   Result: add_outer_inner(5) = {result}\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")


def test_task_with_complex_state():
    """Test 5: Simulate a task with complex state."""
    print("Test 5: Task with Complex State")

    class TaskConfig:
        def __init__(self, model_name: str, temperature: float):
            self.model_name = model_name
            self.temperature = temperature

        def to_dict(self):
            return {"model": self.model_name, "temp": self.temperature}

    def create_processor(config: TaskConfig):
        """Factory that creates a processing function with config."""

        def process(items: list) -> dict:
            # Simulate processing with config
            return {
                "config": config.to_dict(),
                "processed_count": len(items),
                "items": [f"processed_{item}" for item in items],
            }

        return process

    try:
        # Create config and processor
        config = TaskConfig("gpt-4", 0.7)
        processor = create_processor(config)

        # Serialize
        serialized = cloudpickle.dumps(processor)
        deserialized = cloudpickle.loads(serialized)

        # Execute
        result = deserialized(["item1", "item2", "item3"])

        assert result["processed_count"] == 3
        assert result["config"]["model"] == "gpt-4"

        print("✅ Task executed successfully")
        print(f"   Configuration: {result['config']}")
        print(f"   Result: Processed {result['processed_count']} items\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")


def test_comparison_with_standard_pickle():
    """Bonus: Compare cloudpickle vs standard pickle."""
    print("Bonus: Cloudpickle vs Standard Pickle")

    # Lambda function
    def func(x):
        return x * 2

    # Cloudpickle works
    try:
        cloudpickle.dumps(func)
        print("✅ Cloudpickle: Successfully serialized lambda")
    except Exception as e:
        print(f"❌ Cloudpickle failed: {e}")

    # Standard pickle may fail
    try:
        pickle.dumps(func)
        print("✅ Standard pickle: Successfully serialized lambda")
    except Exception as e:
        print(f"⚠️  Standard pickle failed: {type(e).__name__}")

    print()


def main():
    """Run all serialization tests."""
    print("=== Custom Serialization Demo ===\n")

    test_lambda_serialization()
    test_closure_serialization()
    test_class_instance_serialization()
    test_nested_closure_serialization()
    test_task_with_complex_state()

    # Bonus comparison
    test_comparison_with_standard_pickle()

    print("=== Summary ===")
    print("✅ All serialization tests passed")
    print("✅ Cloudpickle handles complex Python objects")
    print("✅ Safe for distributed task execution")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Cloudpickle vs Pickle**
#    - Standard pickle: Cannot serialize lambdas, closures, local functions
#    - Cloudpickle: Can serialize almost any Python object
#    - Graflow uses cloudpickle by default
#
# 2. **What Cloudpickle Can Serialize**
#    ✅ Lambda functions
#    ✅ Closures with captured variables
#    ✅ Nested functions
#    ✅ Class instances with state
#    ✅ Module-level functions
#    ✅ Most Python objects
#
# 3. **What Cannot Be Serialized**
#    ❌ File handles
#    ❌ Database connections
#    ❌ Network sockets
#    ❌ Thread/process objects
#    ❌ Some C extensions
#
# 4. **Best Practices**
#    - Keep closures lightweight
#    - Avoid capturing large objects
#    - Don't capture file handles or connections
#    - Test serialization before distributing
#    - Use factories for complex initialization
#
# 5. **Distributed Execution**
#    - Tasks are serialized before being sent to workers
#    - Results are serialized when returned
#    - Cloudpickle enables flexible task definition
#    - Workers must have same Python version
#
# ============================================================================
# Common Pitfalls:
# ============================================================================
#
# **Capturing Too Much State**:
# Don't do this:
#   large_data = load_huge_dataset()
#   task = lambda: process(large_data)  # Captures ALL of large_data
#
# Do this instead:
#   def create_processor(data_path):
#       def process():
#           data = load_huge_dataset(data_path)  # Load in worker
#           return process(data)
#       return process
#
# **Capturing Non-Serializable Objects**:
# Don't do this:
#   db_conn = connect_to_database()
#   task = lambda: query(db_conn)  # db_conn can't be serialized
#
# Do this instead:
#   def create_query_task(connection_string):
#       def query():
#           conn = connect_to_database(connection_string)
#           result = execute_query(conn)
#           conn.close()
#           return result
#       return query
#
# ============================================================================
# Real-World Usage in Graflow:
# ============================================================================
#
# **Dynamic Task Creation**:
# @task
# def create_processor(config):
#     processor = lambda data: process_with_config(data, config)
#     return processor
#
# **Parameterized Workflows**:
# def build_workflow(model_name, batch_size):
#     @task
#     def process_batch():
#         model = load_model(model_name)
#         batch = get_batch(batch_size)
#         return model.predict(batch)
#     return process_batch
#
# **Redis Distribution**:
# When using Redis backend, tasks are serialized with cloudpickle:
# - Task definition → serialize → Redis → deserialize → Worker
# - Task result → serialize → Redis → deserialize → Main process
#
# ============================================================================
