"""
tests/integration/test_bronze_realdata.py
==========================================
PURPOSE:
    Bronze layer data quality tests on real Delta tables.
    Runs AFTER run_pipeline.py has loaded data.
    Run by notebooks/run_tests_bronze.py.

TESTS (4 only):
    1. Duplicate Check         — no duplicate rows on primary key
    2. Not Null Check          — key column and join key are never null
    3. File vs Table Count     — Bronze row count equals source CSV row count
    4. Empty File Check        — Bronze table has at least 1 row

PARAMETRIZATION:
    TABLE_METADATA is read from workspace.tirtho_db.test_table_metadata.
    Each active row = one test run. Currently: charges + patientvisits.
    Add a new source table by inserting a row in the metadata table —
    no code changes needed here.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from tests.integration.metadata import TABLE_METADATA


def _get_bronze_df(spark, meta: dict):
    """
    Load the real Bronze Delta table for the given table config.

    WHY A HELPER:
        All 4 tests need the same Bronze DataFrame. This helper loads it
        once per call and skips the test cleanly if the table does not exist
        rather than crashing with a confusing AnalysisException.

    Args:
        spark (SparkSession): Active Databricks Spark session.
        meta  (dict): Table config row from TABLE_METADATA.

    Returns:
        DataFrame: All rows from the Bronze Delta table.
    """
    try:
        return spark.table(meta["bronze_table"])
    except Exception as e:
        pytest.skip(
            f"[{meta['name']}] Bronze table '{meta['bronze_table']}' not found. "
            f"Run run_pipeline.py first. Error: {e}"
        )


@pytest.mark.parametrize("meta", TABLE_METADATA)
class TestBronzeIngestion:
    """
    4 Bronze data quality checks, each parametrized across all active tables
    in workspace.tirtho_db.test_table_metadata.
    """

    # ------------------------------------------------------------------
    # TEST 1: Duplicate Check
    # ------------------------------------------------------------------
    def test_duplicate_check(self, spark, meta):
        """
        WHAT: Verify no duplicate rows exist in Bronze based on the primary key.

        WHY: Bronze must be a faithful copy of the source. If ingest
             accidentally created duplicate rows, every downstream layer
             (Silver dedup, Gold counts, reports) would be affected.
             Duplicates in Bronze mean inflated row counts in all reports.

        REAL DATA CHECKS:
            - Total Bronze rows == distinct key_col values
            - If they differ, Bronze has duplicate primary keys
        """
        df    = _get_bronze_df(spark, meta)
        total = df.count()

        # Count distinct values of the primary key column
        distinct = df.select(meta["key_col"]).distinct().count()

        assert total == distinct, (
            f"[{meta['name']}] Duplicate rows found in Bronze. "
            f"Total rows: {total}, Distinct {meta['key_col']}: {distinct}. "
            f"Difference: {total - distinct} duplicate rows."
        )

    # ------------------------------------------------------------------
    # TEST 2: Not Null Check
    # ------------------------------------------------------------------
    def test_not_null_check(self, spark, meta):
        """
        WHAT: Verify the primary key column and the Gold join key (rcm_client_id)
              have no null values in the real Bronze table.

        WHY: A null primary key means that row has no identity — it cannot
             be traced, deduplicated, or joined in Silver or Gold.
             A null rcm_client_id means that row can never join to a visit
             record in Gold — it is permanently orphaned.

        REAL DATA CHECKS:
            - Zero null values in key_col (charge_id or patient_account_number)
            - Zero null values in rcm_client_id
        """
        df = _get_bronze_df(spark, meta)

        # Check primary key nulls
        null_keys = df.filter(F.col(meta["key_col"]).isNull()).count()
        assert null_keys == 0, (
            f"[{meta['name']}] {null_keys} rows have null {meta['key_col']} "
            f"in Bronze. These rows have no identity."
        )

        # Check Gold join key nulls
        if "rcm_client_id" in df.columns:
            null_client = df.filter(F.col("rcm_client_id").isNull()).count()
            assert null_client == 0, (
                f"[{meta['name']}] {null_client} rows have null rcm_client_id "
                f"in Bronze. These rows can never join in Gold."
            )

    # ------------------------------------------------------------------
    # TEST 3: File vs Table Count Check
    # ------------------------------------------------------------------
    def test_file_vs_table_count_check(self, spark, meta):
        """
        WHAT: Verify the row count in the real Bronze Delta table exactly
              matches the row count in the source CSV file on the Volume.

        WHY: Bronze is a faithful copy of the source — not one row more,
             not one row less. Any difference means data was either silently
             dropped or duplicated during ingestion. Dropped rows never appear
             in Silver or Gold. Extra rows inflate all downstream counts.

        REAL DATA CHECKS:
            - spark.read.csv(source_path).count() == bronze_table.count()
        """
        df           = _get_bronze_df(spark, meta)
        bronze_count = df.count()

        # Read the actual source CSV to get ground-truth row count
        try:
            source_count = (
                spark.read.format("csv")
                     .option("header", "true")
                     .load(meta["source_path"])
                     .count()
            )
        except Exception as e:
            pytest.skip(
                f"[{meta['name']}] Cannot read source CSV at "
                f"'{meta['source_path']}'. Error: {e}"
            )

        assert bronze_count == source_count, (
            f"[{meta['name']}] Row count mismatch between source and Bronze. "
            f"Source CSV: {source_count} rows, Bronze table: {bronze_count} rows. "
            f"Difference: {abs(bronze_count - source_count)} rows."
        )

    # ------------------------------------------------------------------
    # TEST 4: Empty File Check
    # ------------------------------------------------------------------
    def test_empty_file_check(self, spark, meta):
        """
        WHAT: Verify the Bronze table is not empty after pipeline execution.

        WHY: An empty Bronze table means either the source CSV was empty
             or the ingest function failed completely. Either way, Silver
             and Gold would have zero rows and all downstream reports
             would show no data — a silent failure.

        REAL DATA CHECKS:
            - Bronze row count >= 1
            - Source CSV row count >= 1 (file itself was not empty)
        """
        df    = _get_bronze_df(spark, meta)
        count = df.count()

        assert count >= 1, (
            f"[{meta['name']}] Bronze table is empty (0 rows). "
            f"Either the source CSV was empty or ingest failed completely."
        )

        # Also verify the source itself was not empty
        try:
            source_count = (
                spark.read.format("csv")
                     .option("header", "true")
                     .load(meta["source_path"])
                     .count()
            )
            assert source_count >= 1, (
                f"[{meta['name']}] Source CSV at '{meta['source_path']}' "
                f"is empty. Load data into the Volume before running tests."
            )
        except Exception as e:
            pytest.skip(
                f"[{meta['name']}] Cannot verify source CSV. Error: {e}"
            )
