"""Comprehensive example showcasing revised graflow features."""

from graflow.core.decorators import task
from graflow.core.task import Task
from graflow.core.workflow import clear_workflow_context, current_workflow_context

print("="*60)
print("GRAFLOW v0.2.0 - Comprehensive Feature Demo")
print("="*60)

# Start fresh
clear_workflow_context()
ctx = current_workflow_context()

print("\n1. DECORATOR-BASED TASKS")
print("-" * 30)

@task
def data_extraction():
    """Extract data from source."""
    print("📊 Extracting data from database...")
    return {"records": 1000}

@task
def data_validation():
    """Validate extracted data."""
    print("✅ Validating data integrity...")
    return {"valid": True}

@task(id="custom_transform")
def data_transformation():
    """Transform data format."""
    print("🔄 Transforming data format...")
    return {"transformed": True}

@task
def generate_report():
    """Generate final report."""
    print("📋 Generating comprehensive report...")
    return {"report_id": "RPT-001"}

print("✅ Created decorated tasks")

print("\n2. TRADITIONAL TASK OBJECTS")
print("-" * 30)

email_notification = Task("email_notification")
archive_data = Task("archive_data")
cleanup = Task("cleanup")

print("✅ Created traditional Task objects")

print("\n3. BUILDING COMPLEX WORKFLOWS")
print("-" * 30)

# Sequential workflow
print("Setting up: data_extraction >> data_validation >> custom_transform")
data_extraction >> data_validation >> data_transformation # type: ignore

# Parallel processing after transformation
print("Setting up parallel: (generate_report | email_notification | archive_data)")
parallel_group = data_transformation >> (generate_report | email_notification | archive_data)

# Final cleanup
print("Setting up: parallel_group >> cleanup")
parallel_group >> cleanup # type: ignore

print("\n4. GRAPH ANALYSIS")
print("-" * 30)
ctx.show_info()

print("\n5. DEPENDENCY VISUALIZATION")
print("-" * 30)
ctx.visualize_dependencies()

print("\n6. WORKFLOW EXECUTION")
print("-" * 30)
print("Executing complete data processing pipeline...")
print()

ctx.execute("data_extraction", max_steps=15)

print("\n7. MIXED OPERATOR DEMO")
print("-" * 30)

clear_workflow_context()
ctx = current_workflow_context()

@task
def start():
    print("🚀 Starting complex workflow")

@task
def process_A():  # noqa: N802
    print("🔧 Processing A")

@task
def process_B():  # noqa: N802
    print("🔧 Processing B")

@task
def process_C():  # noqa: N802
    print("🔧 Processing C")

@task
def merge():
    print("🔗 Merging results")

@task
def finalize():
    print("🎯 Finalizing workflow")

# Complex operator combination
start >> (process_A | process_B | process_C) >> merge >> finalize # type: ignore

print("Complex workflow structure:")
ctx.show_info()

print("\nExecuting complex workflow:")
ctx.execute("start", max_steps=10)

print("\n8. REVERSE DEPENDENCIES")
print("-" * 30)

clear_workflow_context()
ctx = current_workflow_context()

@task
def step1():
    print("Step 1")

@task
def step2():
    print("Step 2")

@task
def step3():
    print("Step 3")

# Using << operator (reverse dependency)
step3 << step2 << step1  # type: ignore # Same as: step1 >> step2 >> step3

print("Reverse dependency setup:")
ctx.visualize_dependencies()

print("\nExecuting reverse-defined workflow:")
ctx.execute("step1", max_steps=5)

print("\n" + "="*60)
print("🎉 GRAFLOW DEMO COMPLETE")
print("="*60)
print("\nKey Features Demonstrated:")
print("✅ @task decorator for function-based tasks")
print("✅ Traditional Task objects")
print("✅ Sequential workflows (>>)")
print("✅ Reverse dependencies (<<)")
print("✅ Parallel groups (|)")
print("✅ Global graph management")
print("✅ Automatic task registration")
print("✅ Complex workflow composition")
print("✅ Cycle detection")
print("✅ Dependency visualization")
print("✅ Robust execution engine")
