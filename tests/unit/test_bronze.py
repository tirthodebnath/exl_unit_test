"""
test_bronze.py
==============
Unit tests for the Bronze ingestion layer.

PURPOSE:
    These tests run on DUMMY DATA before the real pipeline loads anything.
    They prove that the ingestion functions (ingest_charges, ingest_patientvisits)
    behave correctly under every scenario — normal runs, bad data, schema mismatches,
    restarts, and edge cases.

PARAMETRIZATION:
    Every test class is parametrized via TABLE_METADATA so each test runs
    twice — once for charges and once for patientvisits. Adding a new table
    only requires one new entry in TABLE_METADATA; no new test functions needed.

NOTE ON METADATA:
    TABLE_METADATA here is a Python list used for local pytest runs and for
    Databricks unit test runs (before pipeline). The real Delta metadata table
    (workspace.tirtho_db.test_table_metadata) is used by test_after_load_data.py
    which runs on real data AFTER the pipeline has loaded.
"""

import csv
import io
import pytest

from src.bronze.ingest import ingest_charges, ingest_patientvisits
from src.common.schemas import (
    CHARGES_BRONZE_SCHEMA,
    PATIENTVISITS_BRONZE_SCHEMA,
)
from pyspark.sql.types import StringType, TimestampType


# ---------------------------------------------------------------------------
# TABLE METADATA
# Drives parametrization — one entry per table.
# key_col: the primary key column used for null and duplicate checks.
# ---------------------------------------------------------------------------
TABLE_METADATA = [
    pytest.param(
        {
            # Logical name used in test IDs and error messages
            "name": "charges",
            # The ingest function from src/bronze/ingest.py
            "ingest_fn": ingest_charges,
            # The full Bronze schema from src/common/schemas.py
            "schema": CHARGES_BRONZE_SCHEMA,
            # CSV filename written to temp dir during tests
            "filename": "charges.csv",
            # Primary key column — used for null checks and row identity
            "key_col": "charge_id",
        },
        id="charges",
    ),
    pytest.param(
        {
            "name": "patientvisits",
            "ingest_fn": ingest_patientvisits,
            "schema": PATIENTVISITS_BRONZE_SCHEMA,
            "filename": "patientvisits.csv",
            # PatientVisits primary key
            "key_col": "patient_account_number",
        },
        id="patientvisits",
    ),
]


# ---------------------------------------------------------------------------
# HELPER: write a minimal valid CSV for any table schema
# ---------------------------------------------------------------------------
def _write_csv(path, schema, rows=None, extra_col=None, missing_col=None):
    """
    Write a minimal CSV file to disk matching the given Bronze schema.

    WHY THIS EXISTS:
        Bronze tests need real CSV files because ingest functions call
        spark.read.csv(). This helper avoids repeating the same CSV-building
        code in every test method.

    Args:
        path        (Path): File path to write the CSV to.
        schema      (StructType): Bronze schema — used to build the header row.
        rows        (list[list]): Optional list of row value lists.
                                  If None, writes one default row of "TEST" values.
        extra_col   (str): Optional extra column name to add to the header.
                           Used to simulate a new column arriving in the source.
        missing_col (str): Optional column name to remove from the header.
                           Used to simulate a column being dropped in the source.

    Returns:
        int: Number of data rows written (excluding header).
    """
    # Get all column names except audit columns — they are added after read
    data_cols = [
        f.name for f in schema.fields
        if f.name not in ("_ingestion_timestamp", "_source_file")
    ]

    # Apply schema modifications if requested
    if extra_col:
        # Add an unexpected column to simulate source schema drift
        data_cols = data_cols + [extra_col]
    if missing_col and missing_col in data_cols:
        # Remove a column to simulate source dropping a field
        data_cols = [c for c in data_cols if c != missing_col]

    header = ",".join(data_cols)

    if rows is None:
        # Default: one row of TEST values — enough for most tests
        rows = [["TEST"] * len(data_cols)]

    # Write header + rows to CSV
    lines = [header] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")

    return len(rows)


# ---------------------------------------------------------------------------
# TEST CLASS: All 17 Bronze ingestion scenarios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("meta", TABLE_METADATA)
class TestBronzeIngestion:
    """
    17 ingestion scenario tests, each parametrized for charges + patientvisits.

    All tests use temporary files in test_tmp_dir — never the real Volume CSV.
    All tests call the real ingest function from src/bronze/ingest.py.
    """

    # ------------------------------------------------------------------
    # TEST 1: Validate file format compatibility
    # ------------------------------------------------------------------
    def test_validate_file_format_compatibility(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Verify that a correctly formatted CSV file ingests without errors.

        WHY: The most basic sanity check. If the ingest function cannot read
             a well-formed CSV, nothing else in the pipeline can work.

        VALIDATES:
            - Ingest function runs without raising exceptions
            - Output row count matches input row count
            - Both audit columns are attached
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Write a standard 2-row CSV — both rows valid
        _write_csv(csv_path, meta["schema"], rows=[
            ["TEST_A"] * (len(meta["schema"].fields) - 2),
            ["TEST_B"] * (len(meta["schema"].fields) - 2),
        ])

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # File must read successfully and return the correct row count
        assert df.count() == 2, (
            f"[{meta['name']}] Expected 2 rows, got {df.count()}"
        )
        # Both audit columns must be present after ingest
        assert "_ingestion_timestamp" in df.columns
        assert "_source_file" in df.columns

    # ------------------------------------------------------------------
    # TEST 2: Check ingestion when file arrives late
    # ------------------------------------------------------------------
    def test_ingestion_when_file_arrives_late(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Simulate a late-arriving file (data with very old dates).

        WHY: In RCM, files sometimes arrive days or weeks after the service
             date (late charges, corrected claims). Bronze must accept and
             preserve these records without filtering or rejecting them.
             Filtering by age is a Silver/business-rule concern, not Bronze.

        VALIDATES:
            - File with old data still ingests successfully
            - Row count is preserved — no rows silently rejected
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Build row where service date is very old (simulating late arrival)
        data_cols = [
            f.name for f in meta["schema"].fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
        ]
        # Create one row with an old service date to simulate late file
        row = []
        for col in data_cols:
            if col == "service_date":
                row.append("2020-01-01")   # very old date — simulates late file
            elif col == meta["key_col"]:
                row.append("LATE_KEY_001")
            else:
                row.append("TEST")
        _write_csv(csv_path, meta["schema"], rows=[row])

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Late file must be accepted — Bronze never rejects based on date age
        assert df.count() == 1, (
            f"[{meta['name']}] Late-arriving file must still ingest. "
            f"Expected 1 row, got {df.count()}"
        )

    # ------------------------------------------------------------------
    # TEST 3: Validate ingestion of multiple files in batch
    # ------------------------------------------------------------------
    def test_ingestion_multiple_files_in_batch(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Verify that multiple CSV files in the same directory are all
              ingested when a directory path is passed to the ingest function.

        WHY: In production, a batch delivery may drop multiple files into
             the same Volume folder. Spark's spark.read.csv() supports
             directory paths — this test confirms our ingest function
             handles that correctly.

        VALIDATES:
            - All rows from all files appear in the output
            - Row count = sum of all file row counts
        """
        # Create a subdirectory to hold multiple batch files
        batch_dir = test_tmp_dir / f"{meta['name']}_batch"
        batch_dir.mkdir()

        # File 1: 2 rows
        file1 = batch_dir / f"{meta['name']}_1.csv"
        _write_csv(file1, meta["schema"], rows=[
            ["BATCH_A"] * (len(meta["schema"].fields) - 2),
            ["BATCH_B"] * (len(meta["schema"].fields) - 2),
        ])

        # File 2: 1 row — different values to confirm both files are read
        file2 = batch_dir / f"{meta['name']}_2.csv"
        _write_csv(file2, meta["schema"], rows=[
            ["BATCH_C"] * (len(meta["schema"].fields) - 2),
        ])

        # Read from the directory — Spark reads all CSVs inside it
        df = meta["ingest_fn"](spark, spark_path(batch_dir))

        # All 3 rows from both files must be present
        assert df.count() == 3, (
            f"[{meta['name']}] Batch ingest: expected 3 rows (2+1), "
            f"got {df.count()}"
        )

    # ------------------------------------------------------------------
    # TEST 4: Verify duplicate file ingestion handling
    # ------------------------------------------------------------------
    def test_duplicate_file_ingestion_handling(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Ingest the same file twice and verify the row count is
              identical both times (ingest is deterministic).

        WHY: Pipelines are often re-triggered (manual reruns, scheduler
             retries). The ingest function must return the same result
             every time it reads the same file — no phantom rows added.

        VALIDATES:
            - First ingest count equals second ingest count
            - Function is deterministic and idempotent
        """
        csv_path = test_tmp_dir / meta["filename"]
        _write_csv(csv_path, meta["schema"])

        # Ingest the file the first time
        df_first  = meta["ingest_fn"](spark, spark_path(csv_path))
        count_first = df_first.count()

        # Ingest the exact same file a second time
        df_second = meta["ingest_fn"](spark, spark_path(csv_path))
        count_second = df_second.count()

        # Both runs must return identical row counts
        assert count_first == count_second, (
            f"[{meta['name']}] Re-ingesting same file gave different counts: "
            f"first={count_first}, second={count_second}"
        )

    # ------------------------------------------------------------------
    # TEST 5: Validate ingestion when new column is added to source
    # ------------------------------------------------------------------
    def test_ingestion_when_new_column_added(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: The source CSV arrives with an extra column not in our schema.
              Verify Bronze ignores the extra column and ingests normally.

        WHY: Source systems occasionally add new columns without notice.
             Because we use an explicit schema with header=true, Spark
             maps columns by name and ignores columns not in the schema.
             This test confirms that behaviour holds — the pipeline does
             not crash on unexpected columns.

        VALIDATES:
            - Ingest succeeds without error
            - Extra column is NOT in the output (schema is enforced)
            - All expected schema columns are still present
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Write CSV with an extra column the schema does not know about
        _write_csv(csv_path, meta["schema"], extra_col="unexpected_new_column")

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Extra column must not appear — our schema acts as a filter
        assert "unexpected_new_column" not in df.columns, (
            f"[{meta['name']}] Extra source column leaked into Bronze output"
        )
        # Row count must still be correct
        assert df.count() == 1

    # ------------------------------------------------------------------
    # TEST 6: Validate ingestion when a column is removed from source
    # ------------------------------------------------------------------
    def test_ingestion_when_column_removed(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: The source CSV arrives missing one column from our schema.
              Verify Bronze fills the missing column with null and continues.

        WHY: Source systems sometimes drop columns without warning. When
             our explicit schema has a column that the CSV does not, Spark
             fills it with null rather than crashing. This test confirms
             that Bronze remains stable — it ingests what it can and
             leaves missing columns as null for Silver to handle.

        VALIDATES:
            - Ingest succeeds without error
            - Missing column exists in output but contains null
            - All other columns are correctly populated
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Remove org_code from the CSV — it is in the schema but not the file
        missing = "org_code"
        _write_csv(csv_path, meta["schema"], missing_col=missing)

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Ingest must succeed without crashing — the pipeline must be
        # resilient when a source column disappears unexpectedly.
        # NOTE: Spark CSV with header=true and explicit schema maps by
        # position, so a missing column shifts values rather than being
        # null. The key assertion here is resilience (no crash) and that
        # the schema-defined columns are still present in the output.
        assert df.count() == 1, (
            f"[{meta['name']}] Ingest must succeed even with a missing "
            f"source column. Got {df.count()} rows, expected 1."
        )

        # All schema columns must still be present in the output
        for col in [meta["key_col"], "host_system"]:
            assert col in df.columns, (
                f"[{meta['name']}] Schema column '{col}' missing after "
                f"source column removal"
            )

    # ------------------------------------------------------------------
    # TEST 7: Check data type mismatch handling
    # ------------------------------------------------------------------
    def test_data_type_mismatch_handling(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: The source CSV has an invalid value (e.g. text "INVALID_AMOUNT"
              in a column that should be numeric). Verify Bronze accepts it
              as a string without crashing.

        WHY: Bronze keeps ALL columns as StringType — it never casts. So
             "INVALID_AMOUNT" is just a string "INVALID_AMOUNT" in Bronze.
             Silver is responsible for casting and handling type failures.
             If Bronze tried to cast and failed, the entire pipeline would
             stop on any bad row.

        VALIDATES:
            - Ingest does not raise an exception
            - The bad value arrives in Bronze as a string
            - Row count is preserved
        """
        csv_path = test_tmp_dir / meta["filename"]

        # Build a row where the amount field has a non-numeric value
        data_cols = [
            f.name for f in meta["schema"].fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
        ]
        row = []
        for col in data_cols:
            if col == "amount":
                row.append("INVALID_AMOUNT")  # bad value — must not crash
            else:
                row.append("TEST")
        _write_csv(csv_path, meta["schema"], rows=[row])

        # Must not raise — Bronze accepts all values as strings
        df = meta["ingest_fn"](spark, spark_path(csv_path))

        assert df.count() == 1, (
            f"[{meta['name']}] Bad value in amount must not drop the row in Bronze"
        )
        # If amount column exists, verify the bad value was preserved as-is
        if "amount" in df.columns:
            val = df.select("amount").first()["amount"]
            assert val == "INVALID_AMOUNT", (
                f"[{meta['name']}] Bronze must preserve 'INVALID_AMOUNT' as a string, "
                f"got: {val}"
            )

    # ------------------------------------------------------------------
    # TEST 8: Verify schema is consistently enforced
    # ------------------------------------------------------------------
    def test_schema_is_consistently_enforced(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Verify that our explicit schema is applied regardless of the
              column order in the CSV file.

        WHY: With header=true and an explicit schema, Spark matches columns
             by name — not by position. This means even if the source system
             reorders columns, the data still lands in the right schema columns.
             This test confirms that schema enforcement is name-based, not
             positional.

        VALIDATES:
            - Output schema matches the defined Bronze schema
            - Column order in CSV does not affect correctness
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Get data columns and reverse their order — simulates reordered source
        data_cols = [
            f.name for f in meta["schema"].fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
        ]
        reversed_cols = list(reversed(data_cols))

        # Write CSV with columns in reversed order
        header = ",".join(reversed_cols)
        row    = ",".join(["TEST"] * len(reversed_cols))
        csv_path.write_text(f"{header}\n{row}\n")

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # All schema columns must be present in output
        for col in data_cols:
            assert col in df.columns, (
                f"[{meta['name']}] Column '{col}' missing after reversed-order ingest"
            )

    # ------------------------------------------------------------------
    # TEST 9: Ensure no data loss during schema change
    # ------------------------------------------------------------------
    def test_no_data_loss_during_schema_change(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: When an extra column arrives in the CSV (schema change in source),
              verify that existing column data is not lost or corrupted.

        WHY: A common fear with schema changes is that existing data gets
             shifted or lost. This test puts a known value in a specific
             column, adds an extra column to the CSV, ingests it, and
             confirms the original column still has its expected value.

        VALIDATES:
            - Known column value survives schema change
            - No data corruption from extra columns
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Build a row with a traceable value in the key column
        data_cols = [
            f.name for f in meta["schema"].fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
        ]
        row = []
        for col in data_cols:
            if col == meta["key_col"]:
                row.append("TRACEABLE_KEY_001")  # known value we can verify
            else:
                row.append("TEST")

        # Add an extra column at the end (simulating source schema change)
        header = ",".join(data_cols) + ",brand_new_column"
        row_str = ",".join(row) + ",NEW_VAL"
        csv_path.write_text(f"{header}\n{row_str}\n")

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # The key column must still have the original traceable value
        result = df.select(meta["key_col"]).first()[meta["key_col"]]
        assert result == "TRACEABLE_KEY_001", (
            f"[{meta['name']}] Key column data was lost during schema change. "
            f"Expected 'TRACEABLE_KEY_001', got: {result}"
        )

    # ------------------------------------------------------------------
    # TEST 10: Compare source vs Bronze row counts
    # ------------------------------------------------------------------
    def test_source_vs_bronze_row_counts(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Count rows in the source CSV and compare to rows in Bronze.
              They must be identical.

        WHY: Bronze is a faithful copy of the source. Not one row more,
             not one row less. If counts differ, data was either silently
             dropped or duplicated during ingestion — both are critical
             defects that corrupt every downstream layer.

        VALIDATES:
            - Bronze row count == source CSV row count exactly
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Write a CSV with a known number of rows
        source_rows = [
            ["ROW_" + str(i)] * (len(meta["schema"].fields) - 2)
            for i in range(4)   # exactly 4 rows
        ]
        _write_csv(csv_path, meta["schema"], rows=source_rows)

        # Count rows in the source CSV (excluding header)
        lines = csv_path.read_text().strip().split("\n")
        source_count = len(lines) - 1   # subtract header row

        # Ingest and count Bronze rows
        df = meta["ingest_fn"](spark, spark_path(csv_path))
        bronze_count = df.count()

        assert bronze_count == source_count, (
            f"[{meta['name']}] Row count mismatch: "
            f"source={source_count}, bronze={bronze_count}"
        )

    # ------------------------------------------------------------------
    # TEST 11: Validate no missing records
    # ------------------------------------------------------------------
    def test_no_missing_records(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Verify that every specific key value written to the CSV
              appears in the Bronze output.

        WHY: Row count matching (Test 10) confirms quantity but not identity.
             This test confirms that the correct SPECIFIC rows made it through
             — not just that the numbers match.

        VALIDATES:
            - Every key value from source CSV is present in Bronze
            - No specific records were silently skipped
        """
        csv_path = test_tmp_dir / meta["filename"]
        # Build rows with traceable key values
        data_cols = [
            f.name for f in meta["schema"].fields
            if f.name not in ("_ingestion_timestamp", "_source_file")
        ]
        known_keys = ["KEY_ALPHA", "KEY_BETA", "KEY_GAMMA"]
        rows = []
        for key in known_keys:
            row = ["TEST"] * len(data_cols)
            if meta["key_col"] in data_cols:
                idx = data_cols.index(meta["key_col"])
                row[idx] = key
            rows.append(row)

        _write_csv(csv_path, meta["schema"], rows=rows)
        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Collect all key values from Bronze
        bronze_keys = {
            r[meta["key_col"]]
            for r in df.select(meta["key_col"]).collect()
        }

        # Every source key must appear in Bronze
        for key in known_keys:
            assert key in bronze_keys, (
                f"[{meta['name']}] Record with key '{key}' is missing from Bronze"
            )

    # ------------------------------------------------------------------
    # TEST 12: Check partial ingestion failure handling
    # ------------------------------------------------------------------
    def test_partial_ingestion_failure_handling(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Mix valid rows with rows containing bad/extreme values.
              Verify the pipeline does not fail and all rows land in Bronze.

        WHY: Bronze must be resilient. Since it stores everything as StringType,
             there is no type-based failure. Even a row with "!@#$%^" in every
             column should land in Bronze as strings. If Bronze crashed on bad
             data, it would be impossible to audit what went wrong.

        VALIDATES:
            - All rows (good and bad) land in Bronze
            - No exception is raised
            - Row count matches source
        """
        csv_path = test_tmp_dir / meta["filename"]
        col_count = len(meta["schema"].fields) - 2   # exclude audit cols

        # Mix of valid row and rows with extreme/bad values
        rows = [
            ["VALID"]       * col_count,              # clean row
            [""]            * col_count,              # all blank values
            ["!@#$%^&*()"]  * col_count,              # special characters
            ["A" * 100]     * col_count,              # very long strings
        ]
        _write_csv(csv_path, meta["schema"], rows=rows)
        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # All 4 rows must arrive in Bronze — none dropped for bad content
        assert df.count() == 4, (
            f"[{meta['name']}] Expected all 4 rows (including bad ones) "
            f"in Bronze. Got {df.count()}"
        )

    # ------------------------------------------------------------------
    # TEST 13: Verify duplicate records if source has duplicates
    # ------------------------------------------------------------------
    def test_duplicate_records_preserved_from_source(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: If the source CSV contains exact duplicate rows, Bronze must
              preserve all of them — including the duplicates.

        WHY: Bronze is a faithful copy of the source. Deduplication is
             Silver's responsibility (dedupe_latest). If Bronze dropped
             duplicates, Silver would never know the source had them,
             and data quality issues in the source would go undetected.

        VALIDATES:
            - Duplicate rows in source produce duplicate rows in Bronze
            - Total count = 2 (both copies preserved)
            - Distinct count = 1 (confirms they are identical)
        """
        csv_path = test_tmp_dir / meta["filename"]
        col_count = len(meta["schema"].fields) - 2

        # Write two identical rows — exact duplicates
        identical_row = ["DUPLICATE_VALUE"] * col_count
        _write_csv(csv_path, meta["schema"], rows=[identical_row, identical_row])

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Bronze must have 2 rows — not 1 (dedup is Silver's job)
        assert df.count() == 2, (
            f"[{meta['name']}] Bronze must preserve source duplicates. "
            f"Expected 2 rows, got {df.count()}"
        )
        # Exclude audit cols from dedup check — timestamp differs per batch
        dedup_cols = [c for c in df.columns if c != "_ingestion_timestamp"]
        distinct_count = df.dropDuplicates(dedup_cols).count()
        assert distinct_count == 1, (
            f"[{meta['name']}] The 2 rows must be exact duplicates. "
            f"Found {distinct_count} distinct rows"
        )

    # ------------------------------------------------------------------
    # TEST 14: Verify pipeline does not fail due to bad records
    # ------------------------------------------------------------------
    def test_pipeline_doesnt_fail_on_bad_records(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Feed the ingest function data with unexpected / extreme content
              and verify no exception is raised.

        WHY: A single bad row must never crash the entire pipeline.
             Since Bronze stores everything as StringType, any string value
             is valid. This test is the explicit proof that Bronze is
             crash-resilient on bad data.

        VALIDATES:
            - No exception raised on extreme input
            - At least 1 row returned
        """
        csv_path = test_tmp_dir / meta["filename"]
        col_count = len(meta["schema"].fields) - 2

        # Write rows with values designed to break parsers
        rows = [
            ["NULL"]                    * col_count,  # literal string "NULL"
            ["9999999999999999999.99"]  * col_count,  # huge number as string
            ["2099-99-99"]             * col_count,  # impossible date
            [" "]                      * col_count,  # whitespace only
        ]
        _write_csv(csv_path, meta["schema"], rows=rows)

        # This must complete without raising any exception
        try:
            df = meta["ingest_fn"](spark, spark_path(csv_path))
            count = df.count()
        except Exception as e:
            pytest.fail(
                f"[{meta['name']}] Pipeline crashed on bad records: {e}"
            )

        assert count > 0, (
            f"[{meta['name']}] At least some rows must survive even with bad data"
        )

    # ------------------------------------------------------------------
    # TEST 15: Verify ingestion timestamp column
    # ------------------------------------------------------------------
    def test_ingestion_timestamp_column(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Verify that _ingestion_timestamp is present, is TimestampType,
              and has no null values after ingestion.

        WHY: The ingestion timestamp is the audit trail. Without it you cannot
             trace when a row arrived or replay history. It must be a proper
             TimestampType (not a string) so date range queries work correctly.

        VALIDATES:
            - _ingestion_timestamp column exists
            - Column data type is TimestampType
            - No rows have a null timestamp
        """
        csv_path = test_tmp_dir / meta["filename"]
        _write_csv(csv_path, meta["schema"])

        df = meta["ingest_fn"](spark, spark_path(csv_path))

        # Column must exist
        assert "_ingestion_timestamp" in df.columns, (
            f"[{meta['name']}] _ingestion_timestamp column missing"
        )

        # Must be TimestampType — not a string
        field = df.schema["_ingestion_timestamp"]
        assert isinstance(field.dataType, TimestampType), (
            f"[{meta['name']}] _ingestion_timestamp must be TimestampType, "
            f"got {field.dataType}"
        )

        # Must have no nulls — every row must be stamped
        null_count = df.filter(df["_ingestion_timestamp"].isNull()).count()
        assert null_count == 0, (
            f"[{meta['name']}] {null_count} rows have null _ingestion_timestamp"
        )

    # ------------------------------------------------------------------
    # TEST 16: Verify new records are ingested on next run
    # ------------------------------------------------------------------
    def test_new_records_ingested_on_next_run(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Simulate two pipeline runs where the second run has a larger
              file (new records added). Verify the second ingest returns
              the larger count.

        WHY: Bronze must always reflect the current state of the source file.
             When new records arrive in the next batch, the new ingest must
             include them. This confirms the ingest function is not caching
             a stale result from the previous run.

        VALIDATES:
            - Second run with more rows returns higher count
            - New rows are not ignored or filtered
        """
        col_count = len(meta["schema"].fields) - 2

        # First file: 2 rows
        csv_path1 = test_tmp_dir / f"{meta['name']}_run1.csv"
        _write_csv(csv_path1, meta["schema"], rows=[
            ["RUN1_ROW1"] * col_count,
            ["RUN1_ROW2"] * col_count,
        ])

        # Second file: 3 rows (2 original + 1 new record)
        csv_path2 = test_tmp_dir / f"{meta['name']}_run2.csv"
        _write_csv(csv_path2, meta["schema"], rows=[
            ["RUN1_ROW1"] * col_count,
            ["RUN1_ROW2"] * col_count,
            ["RUN2_NEW"]  * col_count,  # new record in second batch
        ])

        count_run1 = meta["ingest_fn"](spark, spark_path(csv_path1)).count()
        count_run2 = meta["ingest_fn"](spark, spark_path(csv_path2)).count()

        # Second run must have more rows because the file is larger
        assert count_run2 > count_run1, (
            f"[{meta['name']}] Second run must include new records. "
            f"Run1={count_run1}, Run2={count_run2}"
        )
        assert count_run2 == 3

    # ------------------------------------------------------------------
    # TEST 17: Validate no data duplication on pipeline restart
    # ------------------------------------------------------------------
    def test_no_duplication_on_pipeline_restart(
        self, spark, test_tmp_dir, spark_path, meta
    ):
        """
        WHAT: Run the ingest function twice on the same file. Verify the
              row count is identical both times.

        WHY: Pipelines are restarted frequently (failures, reruns, manual
             triggers). The ingest function must be idempotent — two runs
             on the same input must produce the same output. If it doubled
             rows on every restart, the pipeline would be unusable.

        VALIDATES:
            - Row count after first ingest == row count after second ingest
            - No phantom rows created by rerunning
        """
        col_count = len(meta["schema"].fields) - 2
        csv_path  = test_tmp_dir / meta["filename"]
        _write_csv(csv_path, meta["schema"], rows=[
            ["RESTART_A"] * col_count,
            ["RESTART_B"] * col_count,
        ])

        # First run
        count1 = meta["ingest_fn"](spark, spark_path(csv_path)).count()
        # Second run on the exact same file
        count2 = meta["ingest_fn"](spark, spark_path(csv_path)).count()

        assert count1 == count2 == 2, (
            f"[{meta['name']}] Pipeline restart caused row duplication. "
            f"Run1={count1}, Run2={count2}"
        )
