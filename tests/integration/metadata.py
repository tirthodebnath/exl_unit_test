"""
metadata.py
===========
PURPOSE:
    Loads TABLE_METADATA from the real Delta metadata table
    (workspace.tirtho_db.test_table_metadata) at pytest collection time.

    This replaces the hardcoded TABLE_METADATA list used in the unit tests.
    Now the metadata lives in a proper Delta table — add a new source table
    by inserting one row into test_table_metadata. No code changes needed.

WHY A SEPARATE FILE:
    TABLE_METADATA must be available as a module-level variable so pytest
    can build parametrize IDs before running any tests. Putting it here
    (rather than in conftest.py) allows all three real-data test files to
    import it with a single clean import statement.

FALLBACK BEHAVIOUR:
    If the metadata table does not exist (e.g. first-time setup, or the
    SQL script has not been run yet), every parametrized test is marked
    with pytest.mark.skip so the test run completes cleanly with a clear
    message rather than crashing with an import error.
"""

import os
import pytest


def _on_databricks() -> bool:
    """
    Check whether we are running inside a Databricks runtime.

    WHY: Integration tests read from real Delta tables which only exist on
    Databricks. On a local machine we skip them rather than fail with a
    confusing Delta/JDBC error.

    Returns:
        bool: True when DATABRICKS_RUNTIME_VERSION env var is set.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


# Fully qualified name of the metadata table in Unity Catalog
METADATA_TABLE = "workspace.tirtho_db.test_table_metadata"


def _load_table_metadata() -> list:
    """
    Read active table configurations from the Delta metadata table.

    HOW IT WORKS:
        1. Gets the active SparkSession (already running on Databricks)
        2. Reads all rows where active = true from the metadata table
        3. Converts each row into a pytest.param dict for parametrization
        4. Returns the list — used directly in @pytest.mark.parametrize

    SKIP BEHAVIOUR:
        If the metadata table is unreachable (table not created yet, or
        running locally), returns a single placeholder param marked with
        pytest.mark.skip. This means all parametrized integration tests
        will show as SKIPPED rather than ERROR in the test report.

    Returns:
        list: List of pytest.param objects, one per active table entry.
    """
    if not _on_databricks():
        # Not on Databricks — skip all integration tests with a clear reason
        return [pytest.param(
            {"name": "unavailable"},
            id="not_on_databricks",
            marks=pytest.mark.skip(
                reason="Integration tests only run on Databricks. "
                       "Use run_tests_on_databricks.py for unit tests locally."
            )
        )]

    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Read only active rows — inactive rows are excluded from all test runs
        rows = (
            spark.table(METADATA_TABLE)
                 .filter("active = true")
                 .collect()
        )

        if not rows:
            # Table exists but has no active rows — skip with explanation
            return [pytest.param(
                {"name": "no_active_tables"},
                id="no_active_entries",
                marks=pytest.mark.skip(
                    reason=f"No active rows found in {METADATA_TABLE}. "
                           f"Set active=true for at least one table."
                )
            )]

        # Convert Delta rows to pytest.param objects.
        # Each param becomes one test ID (e.g. "charges", "patientvisits").
        return [
            pytest.param(
                {
                    # Logical table name — used in test IDs and error messages
                    "name":               row["table_name"],
                    # Primary key column for null and dedup checks
                    "key_col":            row["key_col"],
                    # Full Volume path to the source CSV file
                    "source_path":        row["source_path"],
                    # Fully qualified Bronze Delta table name
                    "bronze_table":       row["bronze_table"],
                    # Fully qualified Silver Delta table name
                    "silver_table":       row["silver_table"],
                    # Expected number of data columns (excluding audit cols)
                    "expected_col_count": row["expected_col_count"],
                },
                id=row["table_name"],
            )
            for row in rows
        ]

    except Exception as e:
        # Metadata table does not exist yet — skip with setup instructions
        return [pytest.param(
            {"name": "metadata_unavailable"},
            id="metadata_table_missing",
            marks=pytest.mark.skip(
                reason=f"Metadata table '{METADATA_TABLE}' not found. "
                       f"Run notebooks/create_metadata_table.sql first. "
                       f"Error: {e}"
            )
        )]


# ---------------------------------------------------------------------------
# Module-level constant — imported by all three integration test files.
# Loaded once at pytest collection time; never re-read during the test run.
# ---------------------------------------------------------------------------
TABLE_METADATA = _load_table_metadata()
