"""Load tasks - data loading to various destinations."""

import csv

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task


@task(inject_context=True)
def load_to_database(context: TaskExecutionContext, table: str = "posts") -> None:
    """Load data to database (simulated).

    Args:
        context: Task execution context
        table: Target table name
    """
    channel = context.get_channel()
    data = channel.get("data")
    records = data["records"]

    print(f"💾 Loading {len(records)} records to database table '{table}'")
    if records:
        print(f"   Sample: {records[0]}")
    print("✅ Database load completed (simulated)")


@task(inject_context=True)
def load_to_csv(context: TaskExecutionContext, path: str = "/tmp/output.csv") -> None:
    """Load data to CSV file.

    Args:
        context: Task execution context
        path: Output CSV file path
    """
    channel = context.get_channel()
    data = channel.get("data")
    records = data["records"]

    if not records:
        print("⚠️  No records to save")
        return

    # Write to CSV file
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"📄 Saved {len(records)} records to {path}")
