"""Modular ETL tasks package."""

from modular_etl.extract_tasks import extract_from_api, extract_from_csv
from modular_etl.load_tasks import load_to_csv, load_to_database
from modular_etl.transform_tasks import filter_invalid_records, normalize_data

__all__ = [  # noqa: RUF022
    # Extract
    "extract_from_api",
    "extract_from_csv",
    # Transform
    "normalize_data",
    "filter_invalid_records",
    # Load
    "load_to_database",
    "load_to_csv",
]
