# Databricks notebook source
"""
test_after_load_data.py
=======================
PURPOSE:
    Validate REAL DATA after the pipeline has loaded it into Delta tables.
    Unlike run_tests_on_databricks.py (which uses dummy data to test code),
    this notebook uses REAL DATA to test that the data itself is correct.

WHEN TO RUN:
    STEP 3 — Run this AFTER run_pipeline.py has completed successfully.
    Sequence:
        1. run_tests_on_databricks.py  (unit tests on dummy data)
        2. run_pipeline.py             (load real data into Delta tables)
        3. THIS NOTEBOOK               (validate real data in Delta tables)

HOW IT WORKS:
    1. Reads table configuration from workspace.tirtho_db.test_table_metadata
       (a Delta table you created using the provided SQL script)
    2. For each ACTIVE table in metadata: runs Volume, Bronze, and Silver checks
    3. Runs Gold join integrity checks
    4. Collects ALL results first — does NOT stop on first failure
    5. Prints a full PASS/FAIL summary at the very end

FAILURE BEHAVIOUR:
    All checks run even if some fail. Summary is printed at the end.
    If any check fails, the final cell raises AssertionError so Databricks
    marks the job as failed.

INVARIANT-BASED TESTING:
    Real data volumes change every run. We cannot hardcode "assert count == 47832".
    Instead we test INVARIANTS — rules that are always true regardless of volume:
        - Silver count <= Bronze count  (filtering only removes rows)
        - Gold distinct charge_ids == Silver count
        - No null charge_ids in Silver
        - All amount_band values are valid strings
    These hold whether you have 5 rows or 5 million rows.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# ---------------------------------------------------------------------------
# Unity Catalog coordinates — where all tables live
# ---------------------------------------------------------------------------
CATALOG = "workspace"       # Unity Catalog name
SCHEMA  = "tirtho_db"       # Schema (database) name

# Metadata table — read table config from here instead of hardcoding
# This table was created using the SQL script provided with the project
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.test_table_metadata"

# Gold output table — tested independently of the per-table loop
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_rcm_summary"

# Minimum acceptable row count for Bronze tables after pipeline runs.
# If Bronze has fewer than this many rows something went wrong with ingest.
MIN_ROW_THRESHOLD = 1

print(f"Metadata table: {METADATA_TABLE}")
print(f"Gold table:     {GOLD_TABLE}")
print(f"Min row threshold: {MIN_ROW_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — import libraries and get Spark session

# COMMAND ----------

import os
import sys
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, TimestampType, DateType, DecimalType

# ---------------------------------------------------------------------------
# Get the active SparkSession (already running in Databricks)
# We do not create a new one — we reuse the cluster's existing session
# ---------------------------------------------------------------------------
spark = SparkSession.builder.getOrCreate()
print(f"Spark version: {spark.version}")

# ---------------------------------------------------------------------------
# Set up src imports by copying repo to a writable path.
# We need src/ imports for schema column counts and type checks.
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = (
    dbutils.notebook.entry_point
           .getDbutils()
           .notebook()
           .getContext()
           .notebookPath()
           .get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]
WORK_DIR  = "/tmp/exl_after_load_work"

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(REPO_PATH, WORK_DIR)
sys.path.insert(0, WORK_DIR)
sys.dont_write_bytecode = True

import src.common.schemas as schemas
print(f"src schemas loaded: {schemas.__file__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results collector — gathers all check outcomes before printing

# COMMAND ----------

# ---------------------------------------------------------------------------
# results: list of dicts accumulating every check result.
# We do NOT raise on failure — we collect everything first so the summary
# shows the complete picture in one read.
# ---------------------------------------------------------------------------
results = []


def run_check(check_name: str, condition: bool,
              actual=None, expected=None, detail: str = ""):
    """
    Record the outcome of a single validation check.

    WHY THIS FUNCTION EXISTS:
        Instead of assert (which stops on first failure), this function
        appends the result to the results list and continues. At the end
        of the notebook we print all results together and only then raise
        if any failed.

    Args:
        check_name (str): Human-readable name shown in the summary.
        condition  (bool): True = PASS, False = FAIL.
        actual     (any): The value we actually observed (shown in FAIL detail).
        expected   (any): The value we expected (shown in FAIL detail).
        detail     (str): Optional extra context for the message.
    """
    if condition:
        # Build a concise pass message showing the actual value
        msg = f"actual={actual}" if actual is not None else ""
        results.append({
            "status":     "PASS",
            "check":      check_name,
            "detail":     msg,
        })
    else:
        # Build a detailed fail message showing expected vs actual
        msg = f"expected={expected}, actual={actual}"
        if detail:
            msg += f" | {detail}"
        results.append({
            "status":     "FAIL",
            "check":      check_name,
            "detail":     msg,
        })


def section(title: str):
    """Print a section header in the notebook output for readability."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Read table metadata from Delta table

# COMMAND ----------

section("READING TABLE METADATA")

# ---------------------------------------------------------------------------
# Read the metadata table that was created with the provided SQL script.
# Only rows where active = true are included in test runs.
# This allows disabling a table from tests without deleting the row.
# ---------------------------------------------------------------------------
try:
    metadata_df = spark.table(METADATA_TABLE).filter("active = true")
    table_configs = metadata_df.collect()
    print(f"Found {len(table_configs)} active table(s) in metadata:")
    for row in table_configs:
        print(f"  → {row['table_name']} | key: {row['key_col']} | "
              f"bronze: {row['bronze_table']} | silver: {row['silver_table']}")
except Exception as e:
    # If metadata table doesn't exist or can't be read, nothing else can run
    raise RuntimeError(
        f"Cannot read metadata table '{METADATA_TABLE}'. "
        f"Run the SQL setup script first. Error: {e}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Per-table checks (Volume + Bronze + Silver)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Loop through every active table in the metadata.
# For each table we run three groups of checks:
#   A) Volume checks     — source CSV file exists and is non-empty
#   B) Bronze checks     — raw ingest table is correct
#   C) Silver checks     — cleaned/typed table is correct
# ---------------------------------------------------------------------------
for config in table_configs:

    table_name   = config["table_name"]       # e.g. "charges"
    key_col      = config["key_col"]          # e.g. "charge_id"
    source_path  = config["source_path"]      # e.g. /Volumes/.../charges.csv
    bronze_table = config["bronze_table"]     # e.g. workspace.tirtho_db.bronze_charges
    silver_table = config["silver_table"]     # e.g. workspace.tirtho_db.silver_charges
    exp_col_count = config["expected_col_count"]  # expected column count excl audit

    section(f"TABLE: {table_name.upper()}")

    # ======================================================================
    # GROUP A: VOLUME CHECKS
    # Check the source CSV file in the Volume BEFORE looking at tables.
    # If the source file has problems, we know the issue is upstream.
    # ======================================================================
    print("\n  [A] Volume / Source File Checks")

    # A1: Source file must exist in the Volume
    # If it does not exist, the pipeline would have failed to ingest
    try:
        files = dbutils.fs.ls(os.path.dirname(source_path))
        file_names = [f.name for f in files]
        file_exists = os.path.basename(source_path) in file_names
    except Exception:
        file_exists = False

    run_check(
        check_name=f"[{table_name}] A1: Source CSV exists in Volume",
        condition=file_exists,
        actual=source_path,
        expected="File to exist",
    )

    # A2: Source file must be non-empty (size > 0 bytes)
    # An empty file would have produced zero Bronze rows
    if file_exists:
        try:
            file_info = [f for f in dbutils.fs.ls(os.path.dirname(source_path))
                         if f.name == os.path.basename(source_path)]
            file_size = file_info[0].size if file_info else 0
            run_check(
                check_name=f"[{table_name}] A2: Source CSV is non-empty",
                condition=file_size > 0,
                actual=f"{file_size} bytes",
                expected="> 0 bytes",
            )
        except Exception as e:
            run_check(
                check_name=f"[{table_name}] A2: Source CSV is non-empty",
                condition=False,
                detail=str(e),
            )

    # ======================================================================
    # GROUP B: BRONZE CHECKS
    # Bronze is a faithful copy of the source. Checks are about preservation:
    # nothing dropped, nothing cast, audit columns present.
    # ======================================================================
    print("\n  [B] Bronze Table Checks")

    # Load the Bronze table — all subsequent B checks use this DataFrame
    try:
        bronze_df = spark.table(bronze_table)
        bronze_count = bronze_df.count()
    except Exception as e:
        run_check(
            check_name=f"[{table_name}] B1: Bronze table is readable",
            condition=False,
            detail=str(e),
        )
        continue   # cannot run any more Bronze checks for this table

    # B1: Bronze must have at least MIN_ROW_THRESHOLD rows
    # Zero rows means the file was empty or ingest completely failed
    run_check(
        check_name=f"[{table_name}] B1: Bronze has >= {MIN_ROW_THRESHOLD} rows",
        condition=bronze_count >= MIN_ROW_THRESHOLD,
        actual=bronze_count,
        expected=f">= {MIN_ROW_THRESHOLD}",
    )

    # B2: All non-audit columns in Bronze must be StringType
    # Bronze never casts — if any column is not StringType, the schema was changed
    non_string_cols = [
        f"{f.name}({f.dataType})"
        for f in bronze_df.schema.fields
        if f.name not in ("_ingestion_timestamp", "_source_file")
        and not isinstance(f.dataType, StringType)
    ]
    run_check(
        check_name=f"[{table_name}] B2: All Bronze data columns are StringType",
        condition=len(non_string_cols) == 0,
        actual=non_string_cols if non_string_cols else "all StringType",
        expected="all StringType",
    )

    # B3: _ingestion_timestamp column must exist and be TimestampType
    # Without this, we cannot trace when data arrived or replay history
    has_ts = "_ingestion_timestamp" in bronze_df.columns
    if has_ts:
        ts_type = bronze_df.schema["_ingestion_timestamp"].dataType
        run_check(
            check_name=f"[{table_name}] B3: _ingestion_timestamp is TimestampType",
            condition=isinstance(ts_type, TimestampType),
            actual=str(ts_type),
            expected="TimestampType",
        )
    else:
        run_check(
            check_name=f"[{table_name}] B3: _ingestion_timestamp column exists",
            condition=False,
            actual="column missing",
            expected="_ingestion_timestamp column",
        )

    # B4: _source_file column must exist and have no nulls
    # Every row must know which file it came from — used for audit trails
    has_sf = "_source_file" in bronze_df.columns
    run_check(
        check_name=f"[{table_name}] B4: _source_file column exists",
        condition=has_sf,
        actual="present" if has_sf else "missing",
        expected="present",
    )
    if has_sf:
        null_sf = bronze_df.filter(F.col("_source_file").isNull()).count()
        run_check(
            check_name=f"[{table_name}] B4: _source_file has no null values",
            condition=null_sf == 0,
            actual=f"{null_sf} nulls",
            expected="0 nulls",
        )

    # B5: No rows should have a null _ingestion_timestamp
    # Every ingested row must be timestamped — null means audit trail is broken
    if has_ts:
        null_ts = bronze_df.filter(F.col("_ingestion_timestamp").isNull()).count()
        run_check(
            check_name=f"[{table_name}] B5: No null _ingestion_timestamp rows",
            condition=null_ts == 0,
            actual=f"{null_ts} nulls",
            expected="0 nulls",
        )

    # ======================================================================
    # GROUP C: SILVER CHECKS
    # Silver applies business rules. Checks verify those rules were enforced
    # on the real data — not just on the dummy data in unit tests.
    # ======================================================================
    print("\n  [C] Silver Table Checks")

    try:
        silver_df    = spark.table(silver_table)
        silver_count = silver_df.count()
    except Exception as e:
        run_check(
            check_name=f"[{table_name}] C1: Silver table is readable",
            condition=False,
            detail=str(e),
        )
        continue

    # C1: Silver must have at least 1 row
    run_check(
        check_name=f"[{table_name}] C1: Silver has >= {MIN_ROW_THRESHOLD} rows",
        condition=silver_count >= MIN_ROW_THRESHOLD,
        actual=silver_count,
        expected=f">= {MIN_ROW_THRESHOLD}",
    )

    # C2: Silver row count must be <= Bronze row count
    # Silver only filters — it never adds rows. If Silver > Bronze, something
    # added rows which should be impossible.
    run_check(
        check_name=f"[{table_name}] C2: Silver count <= Bronze count",
        condition=silver_count <= bronze_count,
        actual=f"silver={silver_count}, bronze={bronze_count}",
        expected="silver <= bronze",
    )

    # C3: Primary key column must have no nulls in Silver
    # The filter step must have removed all null-key rows
    if key_col in silver_df.columns:
        null_keys = silver_df.filter(F.col(key_col).isNull()).count()
        run_check(
            check_name=f"[{table_name}] C3: No null {key_col} in Silver",
            condition=null_keys == 0,
            actual=f"{null_keys} null rows",
            expected="0 null rows",
        )

    # C4: Primary key must be unique in Silver
    # Dedup step must have removed duplicate key values
    if key_col in silver_df.columns:
        distinct_keys = silver_df.select(key_col).distinct().count()
        run_check(
            check_name=f"[{table_name}] C4: {key_col} is unique in Silver",
            condition=distinct_keys == silver_count,
            actual=f"{distinct_keys} distinct / {silver_count} total",
            expected="distinct == total",
        )

    # C5: rcm_client_id must never be null in Silver
    # It is the Gold join key — a null here means the row can never join
    if "rcm_client_id" in silver_df.columns:
        null_client = silver_df.filter(F.col("rcm_client_id").isNull()).count()
        run_check(
            check_name=f"[{table_name}] C5: No null rcm_client_id in Silver",
            condition=null_client == 0,
            actual=f"{null_client} null rows",
            expected="0 null rows",
        )

    # C6: For charges — check specific Silver business rules
    if table_name == "charges":

        # C6a: service_date must be DateType in Silver charges
        # Silver casts it from string — if still string the cast failed
        if "service_date" in silver_df.columns:
            svc_type = silver_df.schema["service_date"].dataType
            run_check(
                check_name="[charges] C6a: service_date is DateType in Silver",
                condition=isinstance(svc_type, DateType),
                actual=str(svc_type),
                expected="DateType",
            )

        # C6b: No negative amounts in Silver
        # The filter step must have removed all negative amounts
        if "amount" in silver_df.columns:
            neg_amounts = silver_df.filter(F.col("amount") < 0).count()
            run_check(
                check_name="[charges] C6b: No negative amounts in Silver",
                condition=neg_amounts == 0,
                actual=f"{neg_amounts} negative rows",
                expected="0 negative rows",
            )

        # C6c: amount_band must only contain known band values
        # If any other value appears, the classify function has a bug
        if "amount_band" in silver_df.columns:
            allowed_bands = {"LOW", "MEDIUM", "HIGH", "JUMBO", "UNKNOWN"}
            actual_bands  = {
                r["amount_band"]
                for r in silver_df.select("amount_band").distinct().collect()
            }
            unknown_bands = actual_bands - allowed_bands
            run_check(
                check_name="[charges] C6c: amount_band only has valid values",
                condition=len(unknown_bands) == 0,
                actual=actual_bands,
                expected=f"subset of {allowed_bands}",
                detail=f"Unexpected bands: {unknown_bands}" if unknown_bands else "",
            )

        # C6d: No rows where posting date is before service date
        # This business rule is enforced in Silver filter
        if "charge_posting_date" in silver_df.columns:
            bad_dates = silver_df.filter(
                F.col("charge_posting_date").isNotNull() &
                (F.col("charge_posting_date") < F.col("service_date"))
            ).count()
            run_check(
                check_name="[charges] C6d: No posting date before service date",
                condition=bad_dates == 0,
                actual=f"{bad_dates} rows with posting < service",
                expected="0 rows",
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Gold table checks

# COMMAND ----------

section("GOLD TABLE CHECKS")

# ---------------------------------------------------------------------------
# Gold checks compare the Gold output against Silver charges.
# Because the join is one-to-many, we always use distinct charge_ids
# in Gold for comparisons — not raw Gold row counts.
# ---------------------------------------------------------------------------

try:
    gold_df         = spark.table(GOLD_TABLE)
    silver_charges  = spark.table(f"{CATALOG}.{SCHEMA}.silver_charges")
    gold_count      = gold_df.count()
    silver_chg_cnt  = silver_charges.count()
    print(f"Gold rows:           {gold_count}")
    print(f"Silver charges rows: {silver_chg_cnt}")
except Exception as e:
    run_check(
        check_name="Gold tables are readable",
        condition=False,
        detail=str(e),
    )
    gold_df = None

if gold_df is not None:

    # G1: Gold must have at least 1 row
    run_check(
        check_name="G1: Gold has >= 1 row",
        condition=gold_count >= MIN_ROW_THRESHOLD,
        actual=gold_count,
        expected=f">= {MIN_ROW_THRESHOLD}",
    )

    # G2: Distinct charge_ids in Gold must equal Silver charges row count
    # If they differ, charges were dropped or invented by the join
    gold_distinct_chg = gold_df.select("charge_id").distinct().count()
    run_check(
        check_name="G2: Distinct charge_ids in Gold == Silver charges count",
        condition=gold_distinct_chg == silver_chg_cnt,
        actual=f"gold_distinct={gold_distinct_chg}, silver={silver_chg_cnt}",
        expected="equal",
    )

    # G3: Total charge amount in Gold (deduped) must match Silver total
    # Dedupe on charge_id first to remove one-to-many fan-out copies
    try:
        silver_total = silver_charges.agg(F.sum("amount")).first()[0]
        gold_total   = (gold_df.dropDuplicates(["charge_id"])
                               .agg(F.sum("amount")).first()[0])
        run_check(
            check_name="G3: Total charge amount in Gold matches Silver",
            condition=silver_total == gold_total,
            actual=f"gold={gold_total}",
            expected=f"silver={silver_total}",
        )
    except Exception as e:
        run_check(
            check_name="G3: Total charge amount in Gold matches Silver",
            condition=False,
            detail=str(e),
        )

    # G4: No charge_ids in Gold that do not exist in Silver
    # The join must never invent phantom charge_ids
    silver_ids = {r["charge_id"]
                  for r in silver_charges.select("charge_id").collect()}
    gold_ids   = {r["charge_id"]
                  for r in gold_df.select("charge_id").distinct().collect()}
    phantom    = gold_ids - silver_ids
    run_check(
        check_name="G4: No new charge_ids introduced in Gold",
        condition=len(phantom) == 0,
        actual=f"{len(phantom)} phantom IDs" if phantom else "none",
        expected="0 phantom IDs",
        detail=str(phantom) if phantom else "",
    )

    # G5: All Silver charge_ids must appear in Gold
    # If any are missing, LEFT JOIN accidentally became INNER JOIN
    dropped = silver_ids - gold_ids
    run_check(
        check_name="G5: No Silver charge_ids dropped from Gold",
        condition=len(dropped) == 0,
        actual=f"{len(dropped)} dropped IDs" if dropped else "none",
        expected="0 dropped IDs",
        detail=str(dropped) if dropped else "",
    )

    # G6: rcm_client_id must never be null in Gold
    # It is the join key from the charges side — always populated
    if "rcm_client_id" in gold_df.columns:
        null_client_gold = gold_df.filter(
            F.col("rcm_client_id").isNull()
        ).count()
        run_check(
            check_name="G6: No null rcm_client_id in Gold",
            condition=null_client_gold == 0,
            actual=f"{null_client_gold} null rows",
            expected="0 null rows",
        )

    # G7: pv_ prefixed visit columns must be present
    # Confirms the join produced the expected output structure
    pv_cols_expected = [
        "pv_patient_account_number",
        "pv_computed_los_days",
        "pv_has_insurance_balance",
    ]
    missing_pv = [c for c in pv_cols_expected if c not in gold_df.columns]
    run_check(
        check_name="G7: pv_ visit columns exist in Gold",
        condition=len(missing_pv) == 0,
        actual=f"missing: {missing_pv}" if missing_pv else "all present",
        expected="all pv_ columns present",
    )

    # G8: amount_band must only have valid values in Gold
    # Confirms the Silver enrichment survived the join unchanged
    if "amount_band" in gold_df.columns:
        allowed_bands  = {"LOW", "MEDIUM", "HIGH", "JUMBO", "UNKNOWN"}
        gold_bands     = {
            r["amount_band"]
            for r in gold_df.select("amount_band").distinct().collect()
        }
        bad_gold_bands = gold_bands - allowed_bands
        run_check(
            check_name="G8: amount_band only has valid values in Gold",
            condition=len(bad_gold_bands) == 0,
            actual=gold_bands,
            expected=f"subset of {allowed_bands}",
            detail=f"Unexpected: {bad_gold_bands}" if bad_gold_bands else "",
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Print full summary

# COMMAND ----------

section("VALIDATION SUMMARY")

# ---------------------------------------------------------------------------
# Print ALL results — pass and fail — in one readable block.
# This gives the full picture rather than stopping at the first failure.
# ---------------------------------------------------------------------------
passed = [r for r in results if r["status"] == "PASS"]
failed = [r for r in results if r["status"] == "FAIL"]

print(f"\nTotal checks: {len(results)}")
print(f"  Passed:     {len(passed)}")
print(f"  Failed:     {len(failed)}")
print()

# Print each result with status icon
for r in results:
    icon = "✓" if r["status"] == "PASS" else "✗"
    print(f"  {icon} {r['status']}  {r['check']}")
    if r["detail"] and r["status"] == "FAIL":
        # Only print detail for failures — keeps output readable
        print(f"          → {r['detail']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Final gate: raise if any check failed

# COMMAND ----------

# ---------------------------------------------------------------------------
# Raise AssertionError if ANY check failed.
# This makes the Databricks job show as FAILED so it is visible in the
# job run history and triggers any alerts configured on job failure.
# The detailed messages above already explain what went wrong.
# ---------------------------------------------------------------------------
if failed:
    failed_names = "\n  ".join(r["check"] for r in failed)
    raise AssertionError(
        f"{len(failed)} validation check(s) FAILED on real data:\n"
        f"  {failed_names}\n\n"
        f"Check the summary above for expected vs actual values."
    )

print("\nALL VALIDATION CHECKS PASSED on real data.")
print("The pipeline output is verified and ready for downstream consumption.")
