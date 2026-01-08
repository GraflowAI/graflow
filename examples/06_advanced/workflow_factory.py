"""
Workflow Factory Pattern
=========================

This example demonstrates the workflow factory pattern - creating reusable
workflow templates that can be instantiated multiple times with different
configurations.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Creating workflow factory functions
2. Parameterized workflow generation
3. Workflow template reuse
4. Multiple instances of the same workflow pattern
5. Workflow composition patterns

Expected Output:
----------------
=== Workflow Factory Pattern ===

Scenario 1: Basic Factory Pattern

Creating ETL pipeline instances...

ETL Pipeline 1:
Name: ETL_Pipeline_1
Tasks: 3
Dependencies: 2

ETL Pipeline 2:
Name: ETL_Pipeline_2
Tasks: 3
Dependencies: 2

Executing ETL_Pipeline_1:
ETL_Pipeline_1: Loading data
ETL_Pipeline_1: Transforming data
ETL_Pipeline_1: Saving data

Executing ETL_Pipeline_2:
ETL_Pipeline_2: Loading data
ETL_Pipeline_2: Transforming data
ETL_Pipeline_2: Saving data

=== Summary ===
✅ Workflow factory pattern demonstrated
✅ Created reusable workflow templates
✅ Multiple instances executed independently
✅ Parameterized workflows working correctly
"""

from typing import Optional

from graflow.core.decorators import task
from graflow.core.workflow import WorkflowContext, workflow


def create_etl_pipeline(name: str, source: str = "default") -> WorkflowContext:
    """Factory function for creating ETL workflow instances.

    Args:
        name: Unique name for this workflow instance
        source: Data source identifier

    Returns:
        WorkflowContext configured with ETL tasks
    """
    ctx = workflow(name)

    with ctx:

        @task
        def load():
            """Load data from source."""
            print(f"{name}: Loading data from {source}")
            return {"source": source, "records": 100}

        @task
        def transform():
            """Transform the loaded data."""
            print(f"{name}: Transforming data")
            return {"transformed": True}

        @task
        def save():
            """Save transformed data."""
            print(f"{name}: Saving data")
            return {"saved": True}

        # Setup pipeline
        load >> transform >> save

    return ctx


def create_ml_pipeline(name: str, model_type: str = "linear", epochs: int = 10) -> WorkflowContext:
    """Factory function for creating ML training workflows.

    Args:
        name: Unique name for this workflow instance
        model_type: Type of model to train
        epochs: Number of training epochs

    Returns:
        WorkflowContext configured with ML tasks
    """
    ctx = workflow(name)

    with ctx:

        @task
        def prepare_data():
            """Prepare training data."""
            print(f"{name}: Preparing data for {model_type} model")
            return {"dataset": "training_data"}

        @task
        def train():
            """Train the model."""
            print(f"{name}: Training {model_type} model for {epochs} epochs")
            return {"model": model_type, "accuracy": 0.95}

        @task
        def evaluate():
            """Evaluate model performance."""
            print(f"{name}: Evaluating {model_type} model")
            return {"metrics": {"accuracy": 0.95, "f1": 0.93}}

        @task
        def save_model():
            """Save trained model."""
            print(f"{name}: Saving {model_type} model")
            return {"saved": True}

        # Setup pipeline
        prepare_data >> train >> evaluate >> save_model

    return ctx


def create_data_validation_pipeline(name: str, rules: Optional[list] = None) -> WorkflowContext:
    """Factory for data validation workflows.

    Args:
        name: Unique name for this workflow instance
        rules: List of validation rules to apply

    Returns:
        WorkflowContext configured with validation tasks
    """
    if rules is None:
        rules = ["not_null", "format_check"]

    ctx = workflow(name)

    with ctx:

        @task
        def load_data():
            """Load data to validate."""
            print(f"{name}: Loading data for validation")
            return {"records": 1000}

        @task
        def apply_rules():
            """Apply validation rules."""
            print(f"{name}: Applying {len(rules)} validation rules")
            print(f"  Rules: {', '.join(rules)}")
            return {"valid": 980, "invalid": 20}

        @task
        def generate_report():
            """Generate validation report."""
            print(f"{name}: Generating validation report")
            return {"report": "validation_complete"}

        # Setup pipeline
        load_data >> apply_rules >> generate_report

    return ctx


def scenario_1_basic_factory():
    """Scenario 1: Basic factory pattern usage."""
    print("=== Scenario 1: Basic Factory Pattern ===\n")
    print("Creating ETL pipeline instances...\n")

    # Create multiple instances of the same workflow pattern
    pipeline1 = create_etl_pipeline("ETL_Pipeline_1", source="database")
    pipeline2 = create_etl_pipeline("ETL_Pipeline_2", source="api")

    # Show workflow info
    print("ETL Pipeline 1:")
    pipeline1.show_info()

    print("\nETL Pipeline 2:")
    pipeline2.show_info()

    # Execute both pipelines
    print("\nExecuting ETL_Pipeline_1:")
    pipeline1.execute("load")

    print("\nExecuting ETL_Pipeline_2:")
    pipeline2.execute("load")


def scenario_2_ml_training_factory():
    """Scenario 2: ML training workflow factory."""
    print("\n=== Scenario 2: ML Training Factory ===\n")
    print("Creating ML training workflows...\n")

    # Create different model training workflows
    linear_model = create_ml_pipeline("linear_regression", model_type="linear", epochs=10)
    neural_net = create_ml_pipeline("neural_network", model_type="neural", epochs=50)

    print("Linear Model Pipeline:")
    linear_model.show_info()

    print("\nNeural Network Pipeline:")
    neural_net.show_info()

    # Execute workflows
    print("\nTraining linear model:")
    linear_model.execute("prepare_data")

    print("\nTraining neural network:")
    neural_net.execute("prepare_data")


def scenario_3_validation_factory():
    """Scenario 3: Data validation workflow factory."""
    print("\n=== Scenario 3: Validation Factory ===\n")
    print("Creating validation workflows...\n")

    # Create validation workflows with different rules
    basic_validation = create_data_validation_pipeline("basic_validation", rules=["not_null", "format_check"])

    advanced_validation = create_data_validation_pipeline(
        "advanced_validation", rules=["not_null", "format_check", "range_check", "business_rules"]
    )

    print("Basic Validation:")
    basic_validation.show_info()

    print("\nAdvanced Validation:")
    advanced_validation.show_info()

    # Execute validations
    print("\nRunning basic validation:")
    basic_validation.execute("load_data")

    print("\nRunning advanced validation:")
    advanced_validation.execute("load_data")


def scenario_4_workflow_composition():
    """Scenario 4: Composing workflows from factories."""
    print("\n=== Scenario 4: Workflow Composition ===\n")
    print("Composing a complex workflow from factories...\n")

    # Create a composite workflow using multiple factories
    with workflow("data_processing_pipeline") as master:

        @task
        def start():
            """Start the master pipeline."""
            print("Master: Starting data processing pipeline")

        @task
        def run_validation():
            """Run validation sub-workflow."""
            print("Master: Running validation")
            validation = create_data_validation_pipeline("validation_step", rules=["not_null", "format_check"])
            validation.execute("load_data")

        @task
        def run_etl():
            """Run ETL sub-workflow."""
            print("Master: Running ETL")
            etl = create_etl_pipeline("etl_step", source="validated_data")
            etl.execute("load")

        @task
        def finish():
            """Finish the master pipeline."""
            print("Master: Pipeline complete")

        # Define master pipeline
        start >> run_validation >> run_etl >> finish

        print("Master Pipeline:")
        master.show_info()

        print("\nExecuting master pipeline:")
        master.execute("start")


def main():
    """Run all workflow factory scenarios."""
    print("=== Workflow Factory Pattern ===\n")

    # Scenario 1: Basic factory
    scenario_1_basic_factory()

    # Scenario 2: ML training factory
    scenario_2_ml_training_factory()

    # Scenario 3: Validation factory
    scenario_3_validation_factory()

    # Scenario 4: Workflow composition
    scenario_4_workflow_composition()

    print("\n=== Summary ===")
    print("✅ Workflow factory pattern demonstrated")
    print("✅ Created reusable workflow templates")
    print("✅ Multiple instances executed independently")
    print("✅ Parameterized workflows working correctly")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Factory Pattern Benefits**
#    - Reusable workflow templates
#    - Parameterized workflow creation
#    - Consistent workflow structure
#    - Easy to test and maintain
#
# 2. **When to Use Workflow Factories**
#    ✅ Multiple instances of similar workflows
#    ✅ Parameterized workflow behavior
#    ✅ Workflow templates for different customers
#    ✅ Standardized workflow patterns
#    ✅ Testing with different configurations
#
# 3. **Factory Function Pattern**
#    def create_workflow(name: str, **params) -> WorkflowContext:
#        ctx = workflow(name)
#        with ctx:
#            # Define tasks using params
#            @task
#            def task1():
#                # Use params here
#                pass
#        return ctx
#
# 4. **Best Practices**
#    - Always use unique workflow names
#    - Pass configuration as parameters
#    - Keep factories focused on one workflow type
#    - Document factory parameters
#    - Return WorkflowContext for flexibility
#
# 5. **Composition Patterns**
#    - Factories can call other factories
#    - Workflows can execute sub-workflows
#    - Build complex pipelines from simple components
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Multi-Customer Processing**:
# for customer in customers:
#     pipeline = create_customer_pipeline(
#         f"customer_{customer.id}",
#         customer_config=customer.config
#     )
#     pipeline.execute()
#
# **A/B Testing**:
# control = create_model_pipeline("control", algorithm="v1")
# variant = create_model_pipeline("variant", algorithm="v2")
# control.execute()
# variant.execute()
# compare_results(control, variant)
#
# **Environment-Specific Workflows**:
# if environment == "prod":
#     pipeline = create_pipeline("prod", validation_strict=True)
# else:
#     pipeline = create_pipeline("dev", validation_strict=False)
#
# **Batch Processing**:
# for batch_id, data in enumerate(batches):
#     pipeline = create_batch_pipeline(
#         f"batch_{batch_id}",
#         data=data,
#         batch_size=100
#     )
#     pipeline.execute()
#
# ============================================================================
# Advanced Patterns:
# ============================================================================
#
# **Abstract Factory**:
# class WorkflowFactory:
#     @staticmethod
#     def create(workflow_type: str, **params):
#         if workflow_type == "etl":
#             return create_etl_pipeline(**params)
#         elif workflow_type == "ml":
#             return create_ml_pipeline(**params)
#         else:
#             raise ValueError(f"Unknown workflow type: {workflow_type}")
#
# **Builder Pattern**:
# class WorkflowBuilder:
#     def __init__(self, name: str):
#         self.name = name
#         self.tasks = []
#
#     def add_load_task(self, source: str):
#         self.tasks.append(("load", source))
#         return self
#
#     def add_transform_task(self, method: str):
#         self.tasks.append(("transform", method))
#         return self
#
#     def build(self) -> WorkflowContext:
#         ctx = workflow(self.name)
#         with ctx:
#             # Create tasks based on self.tasks
#             pass
#         return ctx
#
# # Usage:
# pipeline = (WorkflowBuilder("my_pipeline")
#             .add_load_task("database")
#             .add_transform_task("normalize")
#             .build())
#
# **Configuration-Driven Factory**:
# def create_from_config(config_file: str) -> WorkflowContext:
#     config = load_yaml(config_file)
#     return create_workflow(
#         config["name"],
#         **config["parameters"]
#     )
#
# **Decorator-Based Factory**:
# def workflow_factory(workflow_type: str):
#     def decorator(func):
#         def wrapper(name: str, **params):
#             ctx = workflow(name)
#             with ctx:
#                 func(ctx, **params)
#             return ctx
#         return wrapper
#     return decorator
#
# @workflow_factory("etl")
# def create_etl(ctx, source, destination):
#     @task
#     def load():
#         print(f"Loading from {source}")
#     # ...
#
# ============================================================================
