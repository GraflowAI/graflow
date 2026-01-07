"""Extract tasks - data extraction from various sources."""

import csv
import io

import requests

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task


@task(inject_context=True)
def extract_from_api(context: TaskExecutionContext, endpoint: str = "posts") -> None:
    """Extract data from JSONPlaceholder API.

    Args:
        context: Task execution context
        endpoint: API endpoint (posts, users, comments, etc.)
    """
    url = f"https://jsonplaceholder.typicode.com/{endpoint}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    print(f"📥 Extracted {len(data)} records from API ({endpoint})")

    # Store in channel for next tasks
    channel = context.get_channel()
    channel.set("data", {"source": "api", "records": data, "count": len(data)})


@task(inject_context=True)
def extract_from_csv(context: TaskExecutionContext) -> None:
    """Extract data from embedded CSV data.

    Args:
        context: Task execution context
    """
    # Embedded sample CSV data
    csv_data = """id,title,userId
1,Sample Post 1,1
2,Sample Post 2,2
3,Sample Post 3,1"""

    # Parse CSV
    records = list(csv.DictReader(io.StringIO(csv_data)))

    # Convert numeric fields
    for record in records:
        record["id"] = int(record["id"])
        record["userId"] = int(record["userId"])

    print(f"📄 Extracted {len(records)} records from CSV")

    # Store in channel for next tasks
    channel = context.get_channel()
    channel.set("data", {"source": "csv", "records": records, "count": len(records)})
