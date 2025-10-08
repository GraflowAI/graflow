"""
Nested Workflows
================

This example demonstrates nested workflow contexts, where workflows can be
composed hierarchically with inner workflows running within outer workflows.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Nested workflow contexts
2. Inner and outer workflow scoping
3. Workflow composition and reusability
4. Task organization with hierarchy
5. Modular workflow design

Expected Output:
----------------
=== Nested Workflows Demo ===

Scenario 1: Basic Nested Workflows
📦 Outer Workflow: data_processing
   🔹 Inner Workflow: validation_workflow
      ✅ validate_schema completed
      ✅ validate_values completed
   ✅ Inner validation workflow completed

   🔹 Inner Workflow: transformation_workflow
      ✅ normalize_data completed
      ✅ enrich_data completed
   ✅ Inner transformation workflow completed

✅ Outer workflow completed

Scenario 2: Reusable Workflow Components
🔄 Processing batch 1 with reusable workflow
   ✅ extract_batch_1 completed
   ✅ process_batch_1 completed
🔄 Processing batch 2 with reusable workflow
   ✅ extract_batch_2 completed
   ✅ process_batch_2 completed
✅ All batches processed

=== Summary ===
✅ Nested workflows demonstrated
✅ Workflow composition patterns shown
✅ Modular design achieved
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def scenario_1_basic_nested():
    """Scenario 1: Basic nested workflow demonstration."""
    print("Scenario 1: Basic Nested Workflows")
    print("📦 Outer Workflow: data_processing")

    with workflow("data_processing") as outer:

        @task
        def load_data():
            """Load initial data."""
            print("   📥 Loading data...")
            return {"records": 1000, "status": "loaded"}

        @task
        def run_validation():
            """Run validation as a nested workflow."""
            print("   🔹 Inner Workflow: validation_workflow")

            with workflow("validation_workflow") as inner:

                @task
                def validate_schema():
                    """Validate data schema."""
                    print("      ✅ validate_schema completed")
                    return True

                @task
                def validate_values():
                    """Validate data values."""
                    print("      ✅ validate_values completed")
                    return True

                # Define inner workflow
                validate_schema >> validate_values

                # Execute inner workflow
                inner.execute("validate_schema")

            print("   ✅ Inner validation workflow completed\n")
            return {"validation": "passed"}

        @task
        def run_transformation():
            """Run transformation as a nested workflow."""
            print("   🔹 Inner Workflow: transformation_workflow")

            with workflow("transformation_workflow") as inner:

                @task
                def normalize_data():
                    """Normalize data."""
                    print("      ✅ normalize_data completed")
                    return "normalized"

                @task
                def enrich_data():
                    """Enrich data."""
                    print("      ✅ enrich_data completed")
                    return "enriched"

                # Define inner workflow
                normalize_data >> enrich_data

                # Execute inner workflow
                inner.execute("normalize_data")

            print("   ✅ Inner transformation workflow completed\n")
            return {"transformation": "complete"}

        @task
        def save_results():
            """Save final results."""
            print("✅ Outer workflow completed\n")
            return {"status": "saved"}

        # Define outer workflow
        load_data >> run_validation >> run_transformation >> save_results

        # Execute outer workflow
        outer.execute("load_data")


def scenario_2_reusable_components():
    """Scenario 2: Reusable workflow components."""
    print("Scenario 2: Reusable Workflow Components")

    def create_batch_processor(batch_id: int):
        """Factory function to create a reusable batch processing workflow."""

        @task(id=f"process_batch_{batch_id}")
        def process_batch():
            """Process a single batch using nested workflow."""
            print(f"🔄 Processing batch {batch_id} with reusable workflow")

            with workflow(f"batch_{batch_id}_workflow") as inner:

                @task(id=f"extract_batch_{batch_id}")
                def extract():
                    """Extract batch data."""
                    print(f"   ✅ extract_batch_{batch_id} completed")
                    return f"batch_{batch_id}_data"

                @task(id=f"process_batch_data_{batch_id}")
                def process():
                    """Process batch data."""
                    print(f"   ✅ process_batch_{batch_id} completed")
                    return f"batch_{batch_id}_processed"

                # Define inner workflow
                extract >> process

                # Execute inner workflow
                inner.execute(f"extract_batch_{batch_id}")

            return f"batch_{batch_id}_complete"

        return process_batch

    with workflow("batch_processing") as outer:

        # Create multiple batch processors using the reusable pattern
        batch_1 = create_batch_processor(1)
        batch_2 = create_batch_processor(2)

        @task
        def aggregate():
            """Aggregate results from all batches."""
            print("✅ All batches processed\n")
            return "aggregated"

        # Define workflow: process batches then aggregate
        batch_1 >> aggregate
        batch_2 >> aggregate

        # Execute
        outer.execute("process_batch_1")


def scenario_3_hierarchical_organization():
    """Scenario 3: Hierarchical workflow organization."""
    print("Scenario 3: Hierarchical Organization")

    with workflow("etl_pipeline") as outer:

        @task(inject_context=True)
        def extract_phase(context: TaskExecutionContext):
            """Extract phase with nested extraction tasks."""
            print("📂 Phase 1: Extract")

            with workflow("extract_subworkflow") as inner:

                @task
                def extract_source_a():
                    print("   📥 Extracting from source A")
                    return {"source": "A", "count": 100}

                @task
                def extract_source_b():
                    print("   📥 Extracting from source B")
                    return {"source": "B", "count": 150}

                # Extract in parallel
                inner.execute("extract_source_a")

            print("   ✅ Extract phase completed\n")
            return "extract_complete"

        @task(inject_context=True)
        def transform_phase(context: TaskExecutionContext):
            """Transform phase with nested transformations."""
            print("📂 Phase 2: Transform")

            with workflow("transform_subworkflow") as inner:

                @task
                def clean_data():
                    print("   🧹 Cleaning data")
                    return "clean"

                @task
                def enrich_data():
                    print("   ✨ Enriching data")
                    return "enriched"

                clean_data >> enrich_data
                inner.execute("clean_data")

            print("   ✅ Transform phase completed\n")
            return "transform_complete"

        @task(inject_context=True)
        def load_phase(context: TaskExecutionContext):
            """Load phase with nested loading tasks."""
            print("📂 Phase 3: Load")

            with workflow("load_subworkflow") as inner:

                @task
                def load_to_staging():
                    print("   💾 Loading to staging")
                    return "staged"

                @task
                def load_to_production():
                    print("   💾 Loading to production")
                    return "loaded"

                load_to_staging >> load_to_production
                inner.execute("load_to_staging")

            print("   ✅ Load phase completed\n")
            return "load_complete"

        # Define ETL pipeline
        extract_phase >> transform_phase >> load_phase

        # Execute
        outer.execute("extract_phase")


def main():
    """Run all nested workflow scenarios."""
    print("=== Nested Workflows Demo ===\n")

    # Scenario 1: Basic nested workflows
    scenario_1_basic_nested()

    # Scenario 2: Reusable workflow components
    scenario_2_reusable_components()

    # Scenario 3: Hierarchical organization
    scenario_3_hierarchical_organization()

    print("=== Summary ===")
    print("✅ Nested workflows demonstrated")
    print("✅ Workflow composition patterns shown")
    print("✅ Modular design achieved")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Nested Workflow Contexts**
#    - Workflows can contain other workflows
#    - Each workflow has its own context and scope
#    - Inner workflows execute independently
#
# 2. **Workflow Composition**
#    - Break complex workflows into smaller, manageable pieces
#    - Reuse workflow patterns across different contexts
#    - Organize related tasks hierarchically
#
# 3. **Scoping**
#    - Inner workflow tasks are scoped to inner context
#    - Outer workflow doesn't see inner workflow's tasks directly
#    - Results can be passed between workflows via returns
#
# 4. **Use Cases**
#    ✅ Modular pipeline design
#    ✅ Reusable workflow components
#    ✅ Phase-based organization (extract, transform, load)
#    ✅ Multi-tenant processing with isolation
#
# 5. **Best Practices**
#    - Use nested workflows for logical grouping
#    - Keep nesting levels reasonable (2-3 max)
#    - Pass data explicitly between workflow levels
#    - Name inner workflows descriptively
#
# ============================================================================
# Common Patterns:
# ============================================================================
#
# **Pattern 1: Phase-Based Organization**
#
# with workflow("main_pipeline") as outer:
#     @task
#     def phase_1():
#         with workflow("phase_1_details") as inner:
#             # Detailed tasks for phase 1
#             task_a >> task_b
#             inner.execute("task_a")
#
#     @task
#     def phase_2():
#         with workflow("phase_2_details") as inner:
#             # Detailed tasks for phase 2
#             task_c >> task_d
#             inner.execute("task_c")
#
#     phase_1 >> phase_2
#     outer.execute("phase_1")
#
#
# **Pattern 2: Reusable Workflow Factory**
#
# def create_validation_workflow(dataset_name):
#     @task(id=f"validate_{dataset_name}")
#     def validate():
#         with workflow(f"{dataset_name}_validation") as inner:
#             @task
#             def check_schema():
#                 return validate_schema(dataset_name)
#
#             @task
#             def check_quality():
#                 return check_data_quality(dataset_name)
#
#             check_schema >> check_quality
#             inner.execute("check_schema")
#     return validate
#
# # Use the factory
# validate_customers = create_validation_workflow("customers")
# validate_orders = create_validation_workflow("orders")
#
#
# **Pattern 3: Multi-Tenant Isolation**
#
# def process_tenant(tenant_id):
#     with workflow(f"tenant_{tenant_id}") as inner:
#         @task
#         def load_tenant_data():
#             return load_data(tenant_id)
#
#         @task
#         def process_tenant_data():
#             return process_data(tenant_id)
#
#         load_tenant_data >> process_tenant_data
#         inner.execute("load_tenant_data")
#
# with workflow("multi_tenant") as outer:
#     for tenant_id in tenant_ids:
#         process_tenant(tenant_id)
#
# ============================================================================
# Advanced Topics:
# ============================================================================
#
# **Passing Data Between Workflow Levels**
#
# Data flows through return values and task parameters:
#
# with workflow("outer") as outer_ctx:
#     @task
#     def outer_task():
#         # Inner workflow returns a value
#         with workflow("inner") as inner_ctx:
#             @task
#             def inner_task():
#                 return "inner_result"
#
#             inner_task()
#             inner_ctx.execute("inner_task")
#
#         # Access result via return (not via context)
#         return "outer_used_inner"
#
#
# **Depth Limits**
#
# Practical nesting depth guidelines:
# - 1 level: Simple workflows
# - 2 levels: Most production use cases
# - 3 levels: Complex hierarchies (use sparingly)
# - 4+ levels: Avoid - becomes hard to understand and debug
#
#
# **Error Propagation**
#
# Errors in inner workflows propagate to outer:
#
# @task
# def outer_with_error_handling():
#     try:
#         with workflow("inner") as inner:
#             @task
#             def might_fail():
#                 raise ValueError("Error in inner")
#
#             might_fail()
#             inner.execute("might_fail")
#     except ValueError as e:
#         print(f"Caught error from inner workflow: {e}")
#         # Handle or re-raise
#
# ============================================================================
# Comparison with Other Approaches:
# ============================================================================
#
# **Nested Workflows vs Single Flat Workflow**
#
# Nested:
# ✅ Better organization
# ✅ Reusable components
# ✅ Clear logical grouping
# ❌ More complex structure
# ❌ Harder to visualize
#
# Flat:
# ✅ Simple to understand
# ✅ Easy to visualize
# ❌ Can become cluttered
# ❌ Less reusable
#
# **When to Use Nested Workflows**
#
# Use nested workflows when:
# - You have clear logical phases (ETL phases)
# - You need to reuse workflow patterns
# - You want isolation between components
# - You're processing multiple similar items
#
# Use flat workflows when:
# - Workflow is simple (<10 tasks)
# - No clear sub-groupings
# - Simplicity is more important than reusability
#
# ============================================================================
# Production Considerations:
# ============================================================================
#
# **Monitoring Nested Workflows**
#
# Track execution at each level:
#
# @task(inject_context=True)
# def monitored_nested(context):
#     start_time = time.time()
#
#     with workflow("inner") as inner:
#         # Define and execute inner workflow
#         inner.execute("start")
#
#     elapsed = time.time() - start_time
#     context.get_channel().set("inner_duration", elapsed)
#
#
# **Testing Nested Workflows**
#
# Test inner workflows independently:
#
# def test_inner_workflow():
#     with workflow("test_inner") as ctx:
#         # Define inner workflow
#         task_a >> task_b
#         ctx.execute("task_a")
#
#     # Assert results
#     assert ctx.get_result("task_b") == expected_value
#
#
# **Documentation**
#
# Document the nesting structure:
#
# """
# Workflow Structure:
#
# main_pipeline
# ├── phase_1
# │   ├── extract_subworkflow
# │   │   ├── extract_a
# │   │   └── extract_b
# │   └── validate_subworkflow
# │       ├── validate_schema
# │       └── validate_data
# └── phase_2
#     └── transform_subworkflow
#         ├── clean
#         └── enrich
# """
#
# ============================================================================
