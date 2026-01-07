"""
Modular ETL Pattern
====================

Demonstrates how to organize ETL tasks in separate files for better code organization
and reusability.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Task file separation by responsibility (Extract/Transform/Load)
2. Package-based task organization
3. Task reusability across workflows
4. Clean imports with __init__.py

Expected Output:
----------------
=== Modular ETL Pattern ===

Scenario 1: API → Normalize → Database
📥 Extracted 100 records from API (posts)
🔄 Normalized 100 records
💾 Loading 100 records to database table 'posts'
✅ Pipeline completed

Scenario 2: CSV → Filter → CSV File
📄 Extracted 3 records from CSV
🗑️  Filtered: 3 valid / 0 invalid
📄 Saved 3 records to /tmp/output.csv
✅ Pipeline completed
"""

from modular_etl import (
    extract_from_api,
    extract_from_csv,
    filter_invalid_records,
    load_to_csv,
    load_to_database,
    normalize_data,
)

from graflow.core.workflow import workflow


def scenario_1_api_to_database():
    """Scenario 1: Extract from API, normalize, and load to database."""
    print("\n=== Scenario 1: API → Normalize → Database ===\n")

    with workflow("api_to_database") as wf:
        # Define pipeline: API → normalize → database
        extract_from_api >> normalize_data >> load_to_database  # type: ignore

        # Execute workflow
        wf.execute("extract_from_api")

    print("✅ Pipeline completed\n")


def scenario_2_csv_to_file():
    """Scenario 2: Extract from CSV, filter, and save to file."""
    print("\n=== Scenario 2: CSV → Filter → CSV File ===\n")

    with workflow("csv_to_file") as wf:
        # Define pipeline: CSV → filter → CSV file
        extract_from_csv >> filter_invalid_records >> load_to_csv  # type: ignore

        # Execute workflow
        wf.execute("extract_from_csv")

    print("✅ Pipeline completed\n")


def main():
    """Run all scenarios."""
    print("=== Modular ETL Pattern ===")

    # Run scenarios
    scenario_1_api_to_database()
    scenario_2_csv_to_file()

    print("=" * 50)
    print("Summary:")
    print("✅ Tasks organized in separate files")
    print("✅ Extract tasks: modular_etl/extract_tasks.py")
    print("✅ Transform tasks: modular_etl/transform_tasks.py")
    print("✅ Load tasks: modular_etl/load_tasks.py")
    print("=" * 50)


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **File Separation Pattern**
#    - extract_tasks.py: Data extraction (API, CSV, DB)
#    - transform_tasks.py: Data transformation (normalize, filter, enrich)
#    - load_tasks.py: Data loading (database, files, cloud)
#
# 2. **Benefits**
#    ✅ Better code organization
#    ✅ Tasks are reusable across workflows
#    ✅ Easier to test individual tasks
#    ✅ Team members can work on different files
#    ✅ Clear separation of concerns
#
# 3. **Import Pattern**
#    # Simple imports via __init__.py
#    from modular_etl import extract_from_api, normalize_data
#
#    # Or use specific imports
#    from modular_etl.extract_tasks import extract_from_api
#
# 4. **Task Definition**
#    - Define tasks with @task decorator in their respective files
#    - Tasks can be imported and used in any workflow
#    - No need to re-decorate when importing
#
# 5. **Best Practices**
#    ✅ One responsibility per file (Extract/Transform/Load)
#    ✅ Use __init__.py for convenient imports
#    ✅ Keep tasks independent (no circular dependencies)
#    ✅ Document each task with clear docstrings
#    ✅ Use type hints for better IDE support
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Data Pipeline Organization**:
# project/
#   etl_tasks/
#     extract_tasks.py  # All extraction logic
#     transform_tasks.py  # All transformation logic
#     load_tasks.py  # All loading logic
#   workflows/
#     daily_batch.py  # Uses tasks from etl_tasks/
#     realtime_stream.py  # Reuses same tasks
#
# **Multi-Team Development**:
# - Data Engineers: Work on extract_tasks.py
# - Data Scientists: Work on transform_tasks.py
# - DevOps: Work on load_tasks.py
# - No merge conflicts, clear ownership
#
# **Testing Strategy**:
# # Test individual tasks
# from modular_etl.extract_tasks import extract_from_api
# def test_extract():
#     result = extract_from_api.run(endpoint="posts")
#     assert result["count"] > 0
#
# ============================================================================
