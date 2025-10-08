"""
Task Dependencies Example - Data Flow Between Tasks

This example demonstrates:
- Defining multiple tasks
- Passing data between tasks using return values
- Calling tasks as functions

Expected Output:
    Fetching data...
    Processing data...
    Data processed: 100 records
"""

from graflow.core.decorators import task


@task
def fetch_data():
    """Step 1: Fetch data from a source."""
    print("Fetching data...")
    return {"record_count": 100, "status": "success"}


@task
def process_data(data):
    """Step 2: Process the fetched data."""
    print("Processing data...")
    record_count = data["record_count"]
    result = f"Data processed: {record_count} records"
    print(result)
    return result


def main():
    print("=== Task Dependencies Example ===\n")

    # Execute tasks by calling them as functions
    print("Step 1:")
    data = fetch_data()  # Call task as a function

    print("\nStep 2:")
    result = process_data(data)  # Pass result to next task

    print(f"\nFinal result: {result}")
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
