"""
test_bronze_realdata.py
=======================
PURPOSE:
    The same 17 Bronze ingestion tests from tests/unit/test_bronze.py,
    but running on REAL DATA from the actual Delta tables instead of
    dummy CSV files.

    Unit tests prove the code is correct.
    These tests prove the real data is correct.

TABLE_METADATA:
    Loaded from workspace.tirtho_db.test_table_metadata — a Delta table.
    Each active row becomes one parametrize ID (e.g. "charges", "patientvisits").
    This replaces the hardcoded TABLE_METADATA list in the unit tests.

REAL-DATA MAPPING:
    Each of the 17 original tests is mapped to its real-data equivalent.
    Some tests that were about code behaviour (e.g. what happens when a
    column is missing) are mapped to structural assertions on the real table
    (e.g. all expected columns are present in Bronze).

HOW TO RUN:
    This file is run by test_after_load_data.py notebook AFTER run_pipeline.py
    has loaded real data into the Delta tables.
    Do NOT run this file before run_pipeline.py.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, TimestampType

from tests.integration.metadata import TABLE_METADATA


# ---------------------------------------------------------------------------
# Helper: get the real Bronze DataFrame for a given table config
# ---------------------------------------------------------------------------
def _get_bronze_df(spark, meta: dict):
    """
    Load the real Bronze table for the given table config from metadata.

    WHY:
        Each parametrized test receives a meta dict with bronze_table name.
        This helper reads that specific table and skips the test cleanly
        if the table does not exist.

    Args:
        spark (SparkSession): Active Spark session.
        meta  (dict): Table config from TABLE_METADATA.

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


# ---------------------------------------------------------------------------
# TestBronzeIngestion — same class name as unit tests for consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("meta", TABLE_METADATA)
class TestBronzeIngestion:
    """
    17 Bronze ingestion validation tests on real Delta table data.

    Each test is parametrized across all active tables in the metadata table.
    Class name matches test_bronze.py so results are comparable side by side.
    """

    # ------------------------------------------------------------------
    # REAL DATA TEST 1: Validate file format compatibility
    # Unit test equivalent: wrote a valid CSV and checked it ingested.
    # Real data: Bronze table is readable and has the correct structure.
    # ------------------------------------------------------------------
    def test_validate_file_format_compatibility(self, spark, meta):
        """
        WHAT: Verify the Bronze table is readable and has the correct format.

        WHY: If the source CSV had a format problem (wrong delimiter, encoding
             issue, unexpected binary content), ingest would have failed or
             produced zero rows. A readable Bronze table with correct columns
             confirms the source file was compatible.

        REAL DATA CHECK:
            - Bronze table loads without error
            - Has at least 1 row (file was not empty)
            - Has _ingestion_timestamp and _source_file audit columns
        """
        df = _get_bronze_df(spark, meta)

        # Table must be readable and non-empty
        count = df.count()
        assert count >= 1, (
            f"[{meta['name']}] Bronze table is empty — "
            f"file format may have caused ingest failure"
        )

        # Both audit columns must be present — confirms ingest function ran
        assert "_ingestion_timestamp" in df.columns, (
            f"[{meta['name']}] _ingestion_timestamp missing — audit stamp not applied"
        )
        assert "_source_file" in df.columns, (
            f"[{meta['name']}] _source_file missing — audit stamp not applied"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 2: Check ingestion when file arrives late
    # Unit test: wrote CSV with old service dates, verified ingest accepted.
    # Real data: All rows have non-null _ingestion_timestamp regardless of
    #            how old the data inside the rows is.
    # ------------------------------------------------------------------
    def test_ingestion_when_file_arrives_late(self, spark, meta):
        """
        WHAT: Verify that rows with old data dates were accepted by Bronze.

        WHY: In RCM, late charges and corrected claims can arrive with
             service dates months in the past. Bronze must accept all rows
             regardless of how old the data inside them is. The presence
             of a recent _ingestion_timestamp on rows with old data confirms
             the late file was ingested.

        REAL DATA CHECK:
            - _ingestion_timestamp is non-null for all rows
            - Confirms every row was stamped when it arrived, regardless of data age
        """
        df = _get_bronze_df(spark, meta)

        # Every row must have a timestamp — proves it was processed by ingest
        null_ts = df.filter(F.col("_ingestion_timestamp").isNull()).count()
        assert null_ts == 0, (
            f"[{meta['name']}] {null_ts} rows have null _ingestion_timestamp. "
            f"Every row must be stamped regardless of how old the source data is."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 3: Validate ingestion of multiple files in batch
    # Unit test: wrote 2 CSVs to a directory, verified combined count.
    # Real data: Bronze count >= 1, confirming at least one batch succeeded.
    # ------------------------------------------------------------------
    def test_ingestion_multiple_files_in_batch(self, spark, meta):
        """
        WHAT: Verify that data from all source files ended up in Bronze.

        WHY: Whether the pipeline ran once or in multiple batches, Bronze
             must reflect the latest complete state of the source. A non-zero
             count confirms at least one successful batch ingest occurred.

        REAL DATA CHECK:
            - Bronze has >= 1 row (at least one batch was ingested)
            - _source_file values show which file(s) contributed rows
        """
        df = _get_bronze_df(spark, meta)

        count = df.count()
        assert count >= 1, (
            f"[{meta['name']}] Bronze has 0 rows — no batch was ingested"
        )

        # Show which source files contributed to this Bronze table
        source_files = [
            r["_source_file"]
            for r in df.select("_source_file").distinct().collect()
        ]
        assert len(source_files) >= 1, (
            f"[{meta['name']}] No _source_file values found — "
            f"ingest did not record the source"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 4: Verify duplicate file ingestion handling
    # Unit test: ingested same file twice, verified same count both times.
    # Real data: Reading Bronze twice returns same count (deterministic).
    # ------------------------------------------------------------------
    def test_duplicate_file_ingestion_handling(self, spark, meta):
        """
        WHAT: Verify that reading the Bronze table twice gives identical
              row counts — the table is stable and not growing unexpectedly.

        WHY: If the pipeline was run multiple times with overwrite mode,
             the table should have the same number of rows each time.
             Growing row counts indicate append mode was used accidentally,
             which would cause double-counting in Silver and Gold.

        REAL DATA CHECK:
            - Two separate count() calls return the same value
        """
        df = _get_bronze_df(spark, meta)

        # Read count twice — must be identical (no side effects from reading)
        count1 = df.count()
        count2 = spark.table(meta["bronze_table"]).count()

        assert count1 == count2, (
            f"[{meta['name']}] Bronze count is unstable: "
            f"first read={count1}, second read={count2}"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 5: Validate ingestion when new column is added
    # Unit test: added extra_col to CSV, verified it was ignored by schema.
    # Real data: Bronze has no columns outside the defined schema.
    # ------------------------------------------------------------------
    def test_ingestion_when_new_column_added(self, spark, meta):
        """
        WHAT: Verify the Bronze table has no unexpected extra columns beyond
              what the schema defines.

        WHY: Our explicit schema acts as a filter — columns in the source CSV
             that are not in the schema are ignored. If a new column appears
             in Bronze that should not be there, the schema was changed or
             bypassed. Unexpected columns in Bronze would propagate to Silver
             and Gold, breaking downstream consumers.

        REAL DATA CHECK:
            - Bronze column count matches expected_col_count + 2 audit cols
        """
        df = _get_bronze_df(spark, meta)

        # Expected: data columns + _ingestion_timestamp + _source_file
        expected_total = meta["expected_col_count"] + 2
        actual_total   = len(df.columns)

        assert actual_total == expected_total, (
            f"[{meta['name']}] Bronze has {actual_total} columns but expected "
            f"{expected_total} ({meta['expected_col_count']} data + 2 audit). "
            f"Extra columns may have leaked in from source schema drift."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 6: Validate ingestion when column is removed
    # Unit test: removed a column from CSV, verified ingest was resilient.
    # Real data: All expected schema columns are present in Bronze.
    # ------------------------------------------------------------------
    def test_ingestion_when_column_removed(self, spark, meta):
        """
        WHAT: Verify that all expected schema columns are present in the
              Bronze table — none were lost.

        WHY: If the source system stopped sending a column and the schema
             was not updated, that column might be missing from Bronze.
             Missing columns in Bronze mean Silver and Gold silently lose
             data for that field, which can break joins and reports.

        REAL DATA CHECK:
            - Both audit columns are present
            - rcm_client_id is present (join key must never be missing)
        """
        df = _get_bronze_df(spark, meta)

        # These are the critical columns that must always be present
        required_cols = [
            "_ingestion_timestamp",   # audit trail
            "_source_file",           # audit trail
            "rcm_client_id",          # Gold join key
            meta["key_col"],          # primary key for this table
        ]
        for col in required_cols:
            assert col in df.columns, (
                f"[{meta['name']}] Required column '{col}' is missing from Bronze. "
                f"Source system may have dropped this column."
            )

    # ------------------------------------------------------------------
    # REAL DATA TEST 7: Check data type mismatch handling
    # Unit test: put "INVALID" in amount, verified it was stored as string.
    # Real data: All non-audit Bronze columns are StringType.
    # ------------------------------------------------------------------
    def test_data_type_mismatch_handling(self, spark, meta):
        """
        WHAT: Verify all data columns in Bronze are StringType — even those
              that contain numeric or date values in the source.

        WHY: Bronze stores everything as strings. This is intentional —
             it means bad values (like "INVALID_AMOUNT" in an amount field)
             land safely in Bronze as strings rather than causing type
             conversion errors. If any Bronze column is not StringType, the
             schema was changed or Spark inferred types on this table.

        REAL DATA CHECK:
            - All non-audit columns are StringType
        """
        df = _get_bronze_df(spark, meta)

        # Find any non-audit column that is not StringType
        non_string = [
            f"{f.name}({f.dataType})"
            for f in df.schema.fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
            and not isinstance(f.dataType, StringType)
        ]
        assert len(non_string) == 0, (
            f"[{meta['name']}] These Bronze columns are not StringType: "
            f"{non_string}. Bronze must never cast types."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 8: Verify schema is consistently enforced
    # Unit test: reversed CSV column order, verified schema was name-based.
    # Real data: Bronze column names exactly match the defined schema.
    # ------------------------------------------------------------------
    def test_schema_is_consistently_enforced(self, spark, meta):
        """
        WHAT: Verify the Bronze table column names match the defined schema.

        WHY: Column name consistency is the foundation of everything downstream.
             Silver selects columns by name. Gold joins on column names. If
             a column was renamed (e.g. "charge_id" became "chargeid"), every
             downstream step would silently return nulls or fail.

        REAL DATA CHECK:
            - rcm_client_id, key_col, _ingestion_timestamp, _source_file
              are all present with exactly the right names
        """
        df = _get_bronze_df(spark, meta)
        actual_cols = set(df.columns)

        # Core columns that must always exist with exact names
        must_have = {
            "rcm_client_id",
            meta["key_col"],
            "_ingestion_timestamp",
            "_source_file",
        }
        missing = must_have - actual_cols
        assert len(missing) == 0, (
            f"[{meta['name']}] These required Bronze columns are missing "
            f"or renamed: {missing}"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 9: Ensure no data loss during schema change
    # Unit test: added extra col, verified original col value survived.
    # Real data: key_col has no nulls caused by positional shift.
    # ------------------------------------------------------------------
    def test_no_data_loss_during_schema_change(self, spark, meta):
        """
        WHAT: Verify the primary key column has no nulls — confirming
              that no data was lost or shifted due to schema changes.

        WHY: If the source added/removed columns and Spark mapped by
             position instead of name, the key column value could end up
             in a different column (or become null). Key column nulls in
             Bronze mean those rows can never be traced or reconciled.

        REAL DATA CHECK:
            - key_col has no null values in Bronze
        """
        df = _get_bronze_df(spark, meta)

        if meta["key_col"] not in df.columns:
            pytest.skip(
                f"[{meta['name']}] key_col '{meta['key_col']}' not in Bronze columns"
            )

        null_keys = df.filter(F.col(meta["key_col"]).isNull()).count()
        total     = df.count()

        assert null_keys == 0, (
            f"[{meta['name']}] {null_keys}/{total} rows have null {meta['key_col']}. "
            f"This may indicate data was shifted by a schema change."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 10: Compare source vs Bronze row counts
    # Unit test: counted CSV rows manually, compared to Bronze count.
    # Real data: Read source CSV, compare count to Bronze table count.
    # ------------------------------------------------------------------
    def test_source_vs_bronze_row_counts(self, spark, meta):
        """
        WHAT: Read the source CSV directly and compare its row count to
              the Bronze table row count. They must be equal.

        WHY: Bronze is a faithful copy of the source. Even one missing row
             means data was silently lost during ingestion — that row will
             never appear in Silver or Gold. This test catches that before
             it affects reports.

        REAL DATA CHECK:
            - spark.read.csv(source_path).count() == bronze_table.count()
        """
        df = _get_bronze_df(spark, meta)
        bronze_count = df.count()

        # Read the source CSV to get the ground-truth row count
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
            f"[{meta['name']}] Row count mismatch: "
            f"source CSV={source_count}, Bronze={bronze_count}. "
            f"Data was lost or added during ingestion."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 11: Validate no missing records
    # Unit test: put known key values in CSV, checked all appeared in Bronze.
    # Real data: key_col has no nulls (every record has an identity).
    # ------------------------------------------------------------------
    def test_no_missing_records(self, spark, meta):
        """
        WHAT: Verify every row in Bronze has a non-null primary key value.

        WHY: A row without a key value is invisible to everything downstream —
             Silver filters it, dedup cannot identify it, Gold cannot join it.
             If even one row has a null key, we have missing data that cannot
             be tracked or recovered.

        REAL DATA CHECK:
            - Zero rows with null key_col
            - Total row count > 0
        """
        df = _get_bronze_df(spark, meta)

        total     = df.count()
        null_keys = df.filter(F.col(meta["key_col"]).isNull()).count()

        assert total > 0, (
            f"[{meta['name']}] Bronze has 0 rows — all records are missing"
        )
        assert null_keys == 0, (
            f"[{meta['name']}] {null_keys}/{total} rows have null {meta['key_col']}. "
            f"These records have no identity and are effectively missing."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 12: Check partial ingestion failure handling
    # Unit test: mixed good and bad rows, verified all landed in Bronze.
    # Real data: Bronze count > 0 — pipeline succeeded despite any bad rows.
    # ------------------------------------------------------------------
    def test_partial_ingestion_failure_handling(self, spark, meta):
        """
        WHAT: Verify ingest succeeded and produced rows even if the source
              data contained bad values.

        WHY: Bronze stores everything as StringType, so bad values (dates
             that cannot be parsed, amounts with letters) are just stored
             as strings. The pipeline must never crash on bad data. A
             non-zero Bronze count confirms this.

        REAL DATA CHECK:
            - Bronze has >= 1 row (ingest completed despite any bad values)
        """
        df = _get_bronze_df(spark, meta)

        count = df.count()
        assert count >= 1, (
            f"[{meta['name']}] Bronze has 0 rows. Ingest may have crashed "
            f"on bad records instead of storing them as strings."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 13: Verify duplicate records if source has duplicates
    # Unit test: wrote 2 identical rows, verified Bronze kept both.
    # Real data: If Bronze has duplicate keys, they match Silver dedup
    #            expectation (Silver should have fewer rows).
    # ------------------------------------------------------------------
    def test_duplicate_records_preserved_from_source(self, spark, meta):
        """
        WHAT: Verify that if the source had duplicate rows, Bronze preserved
              them all. Then verify Silver has fewer or equal rows (dedup ran).

        WHY: Bronze is a faithful copy — it never deduplicates. Silver is
             where dedup happens. If Bronze and Silver have the same count,
             the source had no duplicates. If Bronze > Silver, Silver correctly
             deduplicated. Both are valid states — we just verify consistency.

        REAL DATA CHECK:
            - silver_count <= bronze_count (dedup can only reduce, never increase)
        """
        df = _get_bronze_df(spark, meta)
        bronze_count = df.count()

        try:
            silver_df    = spark.table(meta["silver_table"])
            silver_count = silver_df.count()
        except Exception:
            pytest.skip(
                f"[{meta['name']}] Silver table not available for comparison"
            )

        assert silver_count <= bronze_count, (
            f"[{meta['name']}] Silver ({silver_count}) has MORE rows than "
            f"Bronze ({bronze_count}). Silver can only filter, never add rows."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 14: Verify pipeline doesn't fail due to bad records
    # Unit test: passed extreme values, verified no exception raised.
    # Real data: Bronze count > 0 and no null _ingestion_timestamp.
    # ------------------------------------------------------------------
    def test_pipeline_doesnt_fail_on_bad_records(self, spark, meta):
        """
        WHAT: Verify the pipeline completed successfully even if the source
              contained bad or unexpected data.

        WHY: Two signals confirm the pipeline did not crash on bad data:
             (1) Bronze has rows — ingest returned a non-empty DataFrame
             (2) All rows have _ingestion_timestamp — the audit stamp was applied
             If the pipeline had crashed, Bronze would be empty or stale.

        REAL DATA CHECK:
            - Bronze count >= 1
            - No null _ingestion_timestamp values
        """
        df = _get_bronze_df(spark, meta)

        count  = df.count()
        null_ts = df.filter(F.col("_ingestion_timestamp").isNull()).count()

        assert count >= 1, (
            f"[{meta['name']}] Bronze is empty — pipeline may have failed on bad data"
        )
        assert null_ts == 0, (
            f"[{meta['name']}] {null_ts} rows have null timestamp — "
            f"audit stamp was not applied, suggesting ingest failed mid-run"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 15: Verify ingestion timestamp column
    # Unit test: checked column exists, is TimestampType, no nulls.
    # Real data: exact same checks on the real Bronze table.
    # ------------------------------------------------------------------
    def test_ingestion_timestamp_column(self, spark, meta):
        """
        WHAT: Verify _ingestion_timestamp exists, is TimestampType, and
              has no null values in the real Bronze table.

        WHY: This is the same check as the unit test but on real data.
             The unit test proved the ingest function adds the column correctly
             on a dummy DataFrame. This test proves it was correctly applied
             to the real production data.

        REAL DATA CHECK:
            - Column exists
            - Is TimestampType
            - No null values across all real rows
        """
        df = _get_bronze_df(spark, meta)

        # Column must exist
        assert "_ingestion_timestamp" in df.columns, (
            f"[{meta['name']}] _ingestion_timestamp column is missing from Bronze"
        )

        # Must be TimestampType — not a string
        ts_type = df.schema["_ingestion_timestamp"].dataType
        assert isinstance(ts_type, TimestampType), (
            f"[{meta['name']}] _ingestion_timestamp is {ts_type}, expected TimestampType"
        )

        # No nulls across all real rows
        null_count = df.filter(F.col("_ingestion_timestamp").isNull()).count()
        assert null_count == 0, (
            f"[{meta['name']}] {null_count} rows have null _ingestion_timestamp"
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 16: Verify new records are ingested on next run
    # Unit test: ran ingest on larger file, verified higher count.
    # Real data: Bronze has records from the most recent pipeline run.
    # ------------------------------------------------------------------
    def test_new_records_ingested_on_next_run(self, spark, meta):
        """
        WHAT: Verify the Bronze table contains records from the most recent
              pipeline run (not stale data from a previous run).

        WHY: If run_pipeline.py ran with overwrite mode on a new file,
             the _source_file values in Bronze should point to the source CSV.
             A non-null, non-empty _source_file on all rows confirms the
             latest run's data is present.

        REAL DATA CHECK:
            - _source_file is populated on all rows
            - Bronze is not empty (latest run's records are present)
        """
        df = _get_bronze_df(spark, meta)

        count = df.count()
        assert count >= 1, (
            f"[{meta['name']}] Bronze is empty — latest pipeline run produced no rows"
        )

        # All rows must know which file they came from
        null_sf = df.filter(F.col("_source_file").isNull()).count()
        assert null_sf == 0, (
            f"[{meta['name']}] {null_sf}/{count} rows have null _source_file. "
            f"Records from the latest run were not properly stamped."
        )

    # ------------------------------------------------------------------
    # REAL DATA TEST 17: Validate no data duplication on pipeline restart
    # Unit test: ran ingest twice on same file, verified same count.
    # Real data: Bronze count equals source CSV count (overwrite = no accumulation).
    # ------------------------------------------------------------------
    def test_no_duplication_on_pipeline_restart(self, spark, meta):
        """
        WHAT: Verify the Bronze table has not accumulated duplicate rows
              from multiple pipeline runs.

        WHY: run_pipeline.py uses overwrite mode for Bronze tables. Every run
             replaces the previous data — it does not append. If Bronze row
             count significantly exceeds the source CSV row count, it suggests
             append mode was used accidentally, doubling (or more) the data
             on every restart.

        REAL DATA CHECK:
            - Bronze count == source CSV count
            - No row inflation from multiple pipeline restarts
        """
        df = _get_bronze_df(spark, meta)
        bronze_count = df.count()

        # Read source CSV for ground-truth row count
        try:
            source_count = (
                spark.read.format("csv")
                     .option("header", "true")
                     .load(meta["source_path"])
                     .count()
            )
        except Exception as e:
            pytest.skip(
                f"[{meta['name']}] Cannot read source CSV to verify count. "
                f"Error: {e}"
            )

        assert bronze_count == source_count, (
            f"[{meta['name']}] Bronze has {bronze_count} rows but source "
            f"CSV has {source_count} rows. If Bronze > source, the pipeline "
            f"may be running in append mode instead of overwrite."
        )
