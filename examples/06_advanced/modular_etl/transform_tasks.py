"""Transform tasks - data transformation and validation."""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task


@task(inject_context=True)
def normalize_data(context: TaskExecutionContext) -> None:
    """Normalize field names and data types.

    Args:
        context: Task execution context
    """
    channel = context.get_channel()
    data = channel.get("data")
    records = data["records"]

    # Normalize field names (userId -> user_id)
    normalized = []
    for record in records:
        normalized_record = {
            "id": int(record.get("id", 0)),
            "title": str(record.get("title", "")).strip(),
            "user_id": int(record.get("userId") or record.get("user_id", 0)),
        }
        normalized.append(normalized_record)

    print(f"🔄 Normalized {len(normalized)} records")

    # Update channel with normalized data
    channel.set("data", {"records": normalized, "count": len(normalized)})


@task(inject_context=True)
def filter_invalid_records(context: TaskExecutionContext) -> None:
    """Filter out invalid records (empty titles).

    Args:
        context: Task execution context
    """
    channel = context.get_channel()
    data = channel.get("data")
    records = data["records"]

    # Filter out records with empty titles
    valid_records = [r for r in records if r.get("title")]
    invalid_count = len(records) - len(valid_records)

    print(f"🗑️  Filtered: {len(valid_records)} valid / {invalid_count} invalid")

    # Update channel with filtered data
    channel.set("data", {"records": valid_records, "count": len(valid_records)})
