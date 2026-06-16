# Databricks notebook source
"""
test_inventory.py
=================
PURPOSE:
    Complete reference of every test in the exl_unit_test project.
    Lists each test, what it does, which file it lives in, which
    Delta table it reads from, and which notebook runs it.

    Run this notebook any time you want a full picture of the test suite.
    It does NOT run any tests — it only displays the inventory.

TOTAL TESTS:
    Unit tests  (dummy data, before pipeline): 31
    Integration tests (real data, after pipeline): 37
    Grand total: 68

HOW TESTS ARE ORGANISED:
    Unit tests      → tests/unit/        → run_tests_on_databricks.py
    Integration     → tests/integration/ → run_tests_bronze.py
                                           run_tests_silver.py
                                           run_tests_gold.py
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Inventory — exl_unit_test RCM Project
# MAGIC Complete list of all tests, layer by layer.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Build the full test inventory as a list of dicts.
# Each dict = one test row in the final DataFrame.
# ---------------------------------------------------------------------------

# Schema for the inventory DataFrame
schema = StructType([
    StructField("layer",        StringType(), True),  # bronze / silver / gold
    StructField("test_type",    StringType(), True),  # unit / integration
    StructField("run_notebook", StringType(), True),  # which notebook runs this test
    StructField("file",         StringType(), True),  # test file path
    StructField("class",        StringType(), True),  # pytest class name
    StructField("test_name",    StringType(), True),  # test method name
    StructField("table_used",   StringType(), True),  # Delta table(s) read
    StructField("what_it_does", StringType(), True),  # plain language description
])

tests = []

# ---------------------------------------------------------------------------
# BRONZE — UNIT TESTS (17 tests × 2 tables = 34 parametrized executions)
# File:     tests/unit/test_bronze.py
# Notebook: notebooks/run_tests_on_databricks.py
# Data:     dummy CSV files written to /tmp/ — no Delta tables
# ---------------------------------------------------------------------------
BRONZE_UNIT_FILE     = "tests/unit/test_bronze.py"
BRONZE_UNIT_NB       = "run_tests_on_databricks.py"
BRONZE_UNIT_CLASS    = "TestBronzeIngestion"
BRONZE_UNIT_TABLE    = "dummy CSV in /tmp/ (no Delta table)"

bronze_unit_tests = [
    ("test_validate_file_format_compatibility",
     "Writes a valid CSV and ingests it. Checks row count, audit columns present. Confirms source format is compatible."),
    ("test_ingestion_when_file_arrives_late",
     "Writes a CSV with very old service dates. Confirms Bronze accepts it — no date-based rejection. Filtering is Silver's job."),
    ("test_ingestion_multiple_files_in_batch",
     "Writes 2 CSVs to a directory, ingests from the directory path. Confirms Spark reads all files and row count = sum of both files."),
    ("test_duplicate_file_ingestion_handling",
     "Ingests same file twice. Confirms both calls return identical row count — ingest is deterministic, no phantom rows."),
    ("test_ingestion_when_new_column_added",
     "Writes CSV with extra column not in schema. Confirms extra column is ignored — explicit schema acts as filter."),
    ("test_ingestion_when_column_removed",
     "Writes CSV missing a column. Confirms ingest succeeds and all schema columns still present in output."),
    ("test_data_type_mismatch_handling",
     "Puts 'INVALID_AMOUNT' in amount column. Confirms Bronze stores it as string without crashing — no casting in Bronze."),
    ("test_schema_is_consistently_enforced",
     "Reverses CSV column order. Confirms Spark maps by column name not position — output schema always matches definition."),
    ("test_no_data_loss_during_schema_change",
     "Adds extra column to CSV. Confirms original key column value survived — no data shift from schema drift."),
    ("test_source_vs_bronze_row_counts",
     "Counts rows in CSV manually. Confirms Bronze row count exactly matches source — not one row more or less."),
    ("test_no_missing_records",
     "Writes known key values to CSV. Confirms each specific key appears in Bronze — no records skipped."),
    ("test_partial_ingestion_failure_handling",
     "Mixes good rows with bad (special chars, huge numbers). Confirms all rows land in Bronze — no crash on bad data."),
    ("test_duplicate_records_preserved_from_source",
     "Writes 2 identical rows. Confirms Bronze has 2 rows — deduplication is Silver's job, not Bronze."),
    ("test_pipeline_doesnt_fail_on_bad_records",
     "Feeds extreme values (NULL string, impossible dates). Confirms ingest completes without raising any exception."),
    ("test_ingestion_timestamp_column",
     "Confirms _ingestion_timestamp exists, is TimestampType, and has no nulls. Every row must be audit-stamped."),
    ("test_new_records_ingested_on_next_run",
     "Ingests file1 (2 rows) then file2 (3 rows). Confirms second run returns higher count — new records are picked up."),
    ("test_no_duplication_on_pipeline_restart",
     "Ingests same file twice. Confirms row count stays at 2 — overwrite mode means no accumulation from restarts."),
]

for name, desc in bronze_unit_tests:
    tests.append(("bronze", "unit", BRONZE_UNIT_NB, BRONZE_UNIT_FILE,
                  BRONZE_UNIT_CLASS, name, BRONZE_UNIT_TABLE, desc))

# ---------------------------------------------------------------------------
# SILVER — UNIT TESTS (3 tests)
# File:     tests/unit/test_silver_charges.py
# Notebook: notebooks/run_tests_on_databricks.py
# Data:     make_charge() dummy rows in memory — no Delta tables
# ---------------------------------------------------------------------------
SILVER_UNIT_FILE  = "tests/unit/test_silver_charges.py"
SILVER_UNIT_NB    = "run_tests_on_databricks.py"
SILVER_UNIT_TABLE = "dummy DataFrame in memory (no Delta table)"

silver_unit_tests = [
    ("TestSilverServiceDate",
     "test_service_date_is_datetime",
     "Feeds a dummy charge with service_date='2024-01-15' through build_silver_charges(). "
     "Confirms service_date is DateType in output — required for AR aging and date arithmetic."),

    ("TestSilverBusinessRules",
     "test_post_date_not_before_service_date",
     "Feeds one valid charge (posting after service) and one invalid (posting before service). "
     "Confirms invalid row is dropped — a charge cannot be posted before it was rendered."),

    ("TestSilverDeduplication",
     "test_no_exact_duplicate_rows",
     "Feeds same charge_id twice with different timestamps. Confirms only 1 row survives "
     "and it is the latest version — corrected resubmissions handled correctly."),
]

for cls, name, desc in silver_unit_tests:
    tests.append(("silver", "unit", SILVER_UNIT_NB, SILVER_UNIT_FILE,
                  cls, name, SILVER_UNIT_TABLE, desc))

# ---------------------------------------------------------------------------
# GOLD — UNIT TESTS (11 tests)
# File:     tests/unit/test_gold.py
# Notebook: notebooks/run_tests_on_databricks.py
# Data:     fixture rows from conftest.py — no Delta tables
# ---------------------------------------------------------------------------
GOLD_UNIT_FILE  = "tests/unit/test_gold.py"
GOLD_UNIT_NB    = "run_tests_on_databricks.py"
GOLD_UNIT_TABLE = "dummy DataFrames in memory (no Delta table)"

gold_unit_tests = [
    ("TestGoldReconciliation",
     "test_charge_count_matches_silver",
     "Confirms distinct charge_ids in Gold == Silver row count. "
     "Catches charges dropped or invented at the join step."),
    ("TestGoldReconciliation",
     "test_total_charge_amount_matches_silver",
     "Dedupes Gold on charge_id then sums. Confirms Gold total == Silver total. "
     "Any difference means money was lost or invented in the join."),
    ("TestGoldReconciliation",
     "test_no_new_charge_ids_introduced_in_gold",
     "Confirms every charge_id in Gold also exists in Silver. "
     "The join must never invent phantom charge_ids."),
    ("TestGoldReconciliation",
     "test_no_silver_charge_ids_dropped_in_gold",
     "Confirms every Silver charge_id appears in Gold. "
     "If any are missing, LEFT JOIN accidentally became INNER JOIN."),
    ("TestGoldJoinCorrectness",
     "test_left_join_keeps_charge_without_matching_visit",
     "Creates orphan charge (no matching visit). Confirms it appears in Gold "
     "with null pv_ columns — not silently dropped."),
    ("TestGoldJoinCorrectness",
     "test_visit_columns_are_prefixed_with_pv",
     "Confirms pv_patient_account_number, pv_computed_los_days etc. exist. "
     "pv_ prefix prevents column name clashes between charges and visits."),
    ("TestGoldJoinCorrectness",
     "test_rcm_client_id_appears_exactly_once",
     "Confirms rcm_client_id appears exactly once in Gold columns — "
     "not duplicated as rcm_client_id + pv_rcm_client_id."),
    ("TestGoldJoinCorrectness",
     "test_one_to_many_row_count_is_correct",
     "Confirms Gold row count = 9 for 5 charges + 3 visits. "
     "EXL001: 4 charges × 2 visits = 8 rows. EXL002: 1×1 = 1 row."),
    ("TestGoldDataAccuracy",
     "test_amount_band_survives_join",
     "Confirms amount_band on CHG001 is still MEDIUM after the join. "
     "Silver-derived columns must not be altered by the Gold join."),
    ("TestGoldDataAccuracy",
     "test_charge_amount_not_altered_by_join",
     "Confirms CHG001 amount is still 250.00 after the join. "
     "Join must only ADD visit columns, never change charge values."),
    ("TestGoldDataAccuracy",
     "test_service_date_type_preserved_in_gold",
     "Confirms service_date is still DateType in Gold after the join. "
     "Spark joins can coerce types — this confirms it did not happen."),
]

for cls, name, desc in gold_unit_tests:
    tests.append(("gold", "unit", GOLD_UNIT_NB, GOLD_UNIT_FILE,
                  cls, name, GOLD_UNIT_TABLE, desc))

# ---------------------------------------------------------------------------
# BRONZE — INTEGRATION TESTS (4 tests × 2 tables = 8 executions)
# File:     tests/integration/test_bronze_realdata.py
# Notebook: notebooks/run_tests_bronze.py
# Data:     real Delta tables — reads from bronze_charges + bronze_patientvisits
# Metadata: table config read from workspace.tirtho_db.test_table_metadata
# ---------------------------------------------------------------------------
BRONZE_INT_FILE  = "tests/integration/test_bronze_realdata.py"
BRONZE_INT_NB    = "run_tests_bronze.py"
BRONZE_INT_CLASS = "TestBronzeIngestion"
BRONZE_INT_TABLE = "bronze_charges + bronze_patientvisits (from metadata table)"

bronze_int_tests = [
    ("test_duplicate_check",
     "Reads real Bronze table. Total rows must equal distinct key_col count. "
     "Any difference means duplicate primary keys — inflation in all downstream counts."),
    ("test_not_null_check",
     "Reads real Bronze table. Confirms key_col (charge_id / patient_account_number) "
     "and rcm_client_id have zero nulls across all real rows."),
    ("test_file_vs_table_count_check",
     "Reads source CSV from Volume AND Bronze table. Confirms row counts are identical. "
     "Any difference means data was lost or added during ingestion."),
    ("test_empty_file_check",
     "Reads real Bronze table. Confirms row count >= 1. "
     "Zero rows means source CSV was empty or ingest failed completely."),
]

for name, desc in bronze_int_tests:
    tests.append(("bronze", "integration", BRONZE_INT_NB, BRONZE_INT_FILE,
                  BRONZE_INT_CLASS, name, BRONZE_INT_TABLE, desc))

# ---------------------------------------------------------------------------
# SILVER — INTEGRATION TESTS (3 tests)
# File:     tests/integration/test_silver_realdata.py
# Notebook: notebooks/run_tests_silver.py
# Data:     real silver_charges Delta table
# ---------------------------------------------------------------------------
SILVER_INT_FILE  = "tests/integration/test_silver_realdata.py"
SILVER_INT_NB    = "run_tests_silver.py"
SILVER_INT_TABLE = "workspace.tirtho_db.silver_charges"

silver_int_tests = [
    ("TestSilverServiceDate",
     "test_service_date_is_datetime",
     "Reads real silver_charges. Confirms service_date column is DateType "
     "across ALL real rows — not just dummy data. Required for date arithmetic."),
    ("TestSilverBusinessRules",
     "test_post_date_not_before_service_date",
     "Reads real silver_charges. Counts rows where posting < service date. "
     "Must be zero — real source data must not violate this business rule."),
    ("TestSilverDeduplication",
     "test_no_exact_duplicate_rows",
     "Reads real silver_charges. Confirms total rows == distinct charge_ids. "
     "Any duplicate means dedup_latest() did not run correctly on real data."),
]

for cls, name, desc in silver_int_tests:
    tests.append(("silver", "integration", SILVER_INT_NB, SILVER_INT_FILE,
                  cls, name, SILVER_INT_TABLE, desc))

# ---------------------------------------------------------------------------
# GOLD — INTEGRATION TESTS: test_charges.py (6 tests)
# File:     tests/integration/gold/test_charges.py
# Notebook: notebooks/run_tests_gold.py
# Data:     gold_rcm_summary_v2 + silver_charges
# ---------------------------------------------------------------------------
GOLD_CHG_FILE  = "tests/integration/gold/test_charges.py"
GOLD_CHG_NB    = "run_tests_gold.py"
GOLD_CHG_CLASS = "TestGoldChargeReconciliation"
GOLD_CHG_TABLE = "workspace.tirtho_db.gold_rcm_summary_v2 + silver_charges"

gold_charges_tests = [
    ("test_charge_count_matches_silver",
     "Reads gold_rcm_summary_v2 and silver_charges. "
     "Distinct charge_ids in Gold must equal Silver row count on real data."),
    ("test_total_charge_amount_matches_silver",
     "Dedupes Gold on charge_id then sums amounts. Confirms total matches Silver. "
     "Catches any money lost or invented by the V2 join on real data."),
    ("test_no_new_charge_ids_in_gold",
     "Reads both tables. Confirms Gold charge_ids ⊆ Silver charge_ids. "
     "The V2 join must never invent phantom charges on real data."),
    ("test_no_silver_charge_ids_dropped",
     "Reads both tables. Confirms Silver charge_ids ⊆ Gold charge_ids. "
     "LEFT JOIN guarantee — no real charge should be silently dropped."),
    ("test_amount_band_valid_in_gold",
     "Reads gold_rcm_summary_v2. Confirms all distinct amount_band values "
     "are in {LOW, MEDIUM, HIGH, JUMBO, UNKNOWN} on real data."),
    ("test_service_date_is_date_type_in_gold",
     "Reads gold_rcm_summary_v2. Confirms service_date column is DateType "
     "in the real Gold table — type survived the V2 join."),
]

for name, desc in gold_charges_tests:
    tests.append(("gold", "integration", GOLD_CHG_NB, GOLD_CHG_FILE,
                  GOLD_CHG_CLASS, name, GOLD_CHG_TABLE, desc))

# ---------------------------------------------------------------------------
# GOLD — INTEGRATION TESTS: test_patientvisits.py (6 tests)
# File:     tests/integration/gold/test_patientvisits.py
# Notebook: notebooks/run_tests_gold.py
# Data:     gold_rcm_summary_v2 + silver_charges
# ---------------------------------------------------------------------------
GOLD_PV_FILE  = "tests/integration/gold/test_patientvisits.py"
GOLD_PV_NB    = "run_tests_gold.py"
GOLD_PV_CLASS = "TestGoldVisitJoinCorrectness"
GOLD_PV_TABLE = "workspace.tirtho_db.gold_rcm_summary_v2 + silver_charges"

gold_visit_tests = [
    ("test_discharge_date_not_null_where_visit_joined",
     "Reads gold_rcm_summary_v2. Where pv_ columns are populated, "
     "pv_patient_discharge_date must never be null. Verifies AND discharge IS NOT NULL condition."),
    ("test_npi_matches_between_charge_and_visit",
     "Reads gold_rcm_summary_v2. Where visit joined, rcm_npi must equal pv_rcm_npi. "
     "Verifies AND rcm_npi = rcm_npi join condition on real data."),
    ("test_patient_account_number_matches",
     "Reads gold_rcm_summary_v2. Where visit joined, patient_account_number "
     "must equal pv_patient_account_number. Verifies third join condition on real data."),
    ("test_visit_columns_prefixed_pv",
     "Reads gold_rcm_summary_v2. Confirms pv_patient_account_number, "
     "pv_rcm_npi, pv_patient_discharge_date all exist — pv_ prefix applied correctly."),
    ("test_left_join_keeps_all_charges",
     "Reads gold_rcm_summary_v2 and silver_charges. "
     "Distinct charge_ids in Gold >= Silver count. LEFT JOIN must keep all charges."),
    ("test_rcm_client_id_not_null_in_gold",
     "Reads gold_rcm_summary_v2. Confirms rcm_client_id is never null. "
     "Comes from charges side — all charges have a client ID after Silver filter."),
]

for name, desc in gold_visit_tests:
    tests.append(("gold", "integration", GOLD_PV_NB, GOLD_PV_FILE,
                  GOLD_PV_CLASS, name, GOLD_PV_TABLE, desc))

# ---------------------------------------------------------------------------
# GOLD — INTEGRATION TESTS: test_ogom_charges.py (14 tests)
# File:     tests/integration/gold/test_ogom_charges.py
# Notebook: notebooks/run_tests_gold.py
# Data:     gold_ogom_charges + silver_charges
# ---------------------------------------------------------------------------
GOLD_OGOM_FILE  = "tests/integration/gold/test_ogom_charges.py"
GOLD_OGOM_NB    = "run_tests_gold.py"
GOLD_OGOM_TABLE = "workspace.tirtho_db.gold_ogom_charges + silver_charges"

gold_ogom_tests = [
    ("TestOGOMChargesIdentity",
     "test_charge_id_not_null",
     "Reads gold_ogom_charges. Confirms charge_id never null. "
     "A null charge_id means an untraceable row — cannot link to patient, claim, or payer."),
    ("TestOGOMChargesIdentity",
     "test_rcm_client_id_not_null",
     "Reads gold_ogom_charges. Confirms rcm_client_id never null. "
     "Null client_id makes row invisible to any client-specific report."),
    ("TestOGOMChargesAmounts",
     "test_charge_amount_not_negative",
     "Reads gold_ogom_charges. Confirms no negative charge_amount. "
     "Silver filter removed negatives — any negative here means filter did not apply."),
    ("TestOGOMChargesAmounts",
     "test_charge_amount_not_null",
     "Reads gold_ogom_charges. Confirms no null charge_amount. "
     "A charge with no amount has no financial value — Silver should have removed these."),
    ("TestOGOMChargesComputedColumns",
     "test_ogom_transaction_type_always_charge",
     "Reads gold_ogom_charges. Confirms ogom_transaction_type is 'Charge' on every row. "
     "Maps to literal 'Charge' AS OGOMTransactionType in the original SQL."),
    ("TestOGOMChargesComputedColumns",
     "test_charge_age_null_when_no_discharge",
     "Reads gold_ogom_charges. Confirms charge_age is null when discharge_date is null. "
     "Maps to CASE WHEN discharge IS NULL THEN NULL in the SQL."),
    ("TestOGOMChargesComputedColumns",
     "test_charge_age_computed_when_discharged",
     "Reads gold_ogom_charges. Confirms charge_age is not null when discharge_date exists. "
     "DATEDIFF(posting_date, discharge_date) must be computed for all discharged patients."),
    ("TestOGOMChargesComputedColumns",
     "test_late_charge_flag_null_when_no_discharge",
     "Reads gold_ogom_charges. Confirms late_charge_flag is null when discharge is null. "
     "Cannot determine if charge is late without knowing discharge date."),
    ("TestOGOMChargesComputedColumns",
     "test_late_charge_flag_values_are_valid",
     "Reads gold_ogom_charges. Confirms late_charge_flag is only 0, 1, or null. "
     "The CASE expression produces no other values."),
    ("TestOGOMChargesComputedColumns",
     "test_charge_lag_days_computed_when_dates_exist",
     "Reads gold_ogom_charges. Confirms charge_lag_days not null when "
     "posting_date AND service_date both exist. Maps to DATEDIFF(posting, service)."),
    ("TestOGOMChargesComputedColumns",
     "test_charge_capture_days_computed_when_dates_exist",
     "Reads gold_ogom_charges. Confirms charge_capture_days not null when "
     "posting_date AND admit_date both exist. Maps to DATEDIFF(posting, admission)."),
    ("TestOGOMChargesJoinIntegrity",
     "test_no_charges_dropped_from_silver",
     "Reads gold_ogom_charges + silver_charges. Distinct charge_ids in OGOM Gold "
     "must equal Silver count. All charges preserved — LEFT JOIN guarantee."),
    ("TestOGOMChargesJoinIntegrity",
     "test_discharge_date_not_null_where_visit_joined",
     "Reads gold_ogom_charges. Where admit_date is populated (visit joined), "
     "discharge_date must not be null. Verifies discharge IS NOT NULL join condition."),
    ("TestOGOMChargesJoinIntegrity",
     "test_gold_has_at_least_one_row",
     "Reads gold_ogom_charges. Confirms row count >= 1. "
     "An empty OGOM Gold table means the join and compute logic failed completely."),
]

for cls, name, desc in gold_ogom_tests:
    tests.append(("gold", "integration", GOLD_OGOM_NB, GOLD_OGOM_FILE,
                  cls, name, GOLD_OGOM_TABLE, desc))

# ---------------------------------------------------------------------------
# Build Spark DataFrame from the complete test list
# ---------------------------------------------------------------------------
inventory_df = spark.createDataFrame(tests, schema=schema)

print(f"Total tests in inventory: {inventory_df.count()}")
print(f"  Unit tests:        {inventory_df.filter('test_type = \"unit\"').count()}")
print(f"  Integration tests: {inventory_df.filter('test_type = \"integration\"').count()}")

# COMMAND ----------

# MAGIC %md ## Bronze Layer Tests

# COMMAND ----------

print("=" * 70)
print("BRONZE LAYER TESTS")
print("=" * 70)

bronze = inventory_df.filter("layer = 'bronze'").collect()

for test_type in ["unit", "integration"]:
    rows = [r for r in bronze if r["test_type"] == test_type]
    if not rows:
        continue
    print(f"\n{'─'*70}")
    print(f"  TYPE:     {test_type.upper()}")
    print(f"  FILE:     {rows[0]['file']}")
    print(f"  NOTEBOOK: {rows[0]['run_notebook']}")
    print(f"  TABLE:    {rows[0]['table_used']}")
    print(f"  TESTS:    {len(rows)}")
    print(f"{'─'*70}")
    for i, r in enumerate(rows, 1):
        print(f"\n  {i}. [{r['class']}]")
        print(f"     {r['test_name']}")
        print(f"     → {r['what_it_does']}")

# COMMAND ----------

# MAGIC %md ## Silver Layer Tests

# COMMAND ----------

print("=" * 70)
print("SILVER LAYER TESTS")
print("=" * 70)

silver = inventory_df.filter("layer = 'silver'").collect()

for test_type in ["unit", "integration"]:
    rows = [r for r in silver if r["test_type"] == test_type]
    if not rows:
        continue
    print(f"\n{'─'*70}")
    print(f"  TYPE:     {test_type.upper()}")
    print(f"  FILE:     {rows[0]['file']}")
    print(f"  NOTEBOOK: {rows[0]['run_notebook']}")
    print(f"  TABLE:    {rows[0]['table_used']}")
    print(f"  TESTS:    {len(rows)}")
    print(f"{'─'*70}")
    for i, r in enumerate(rows, 1):
        print(f"\n  {i}. [{r['class']}]")
        print(f"     {r['test_name']}")
        print(f"     → {r['what_it_does']}")

# COMMAND ----------

# MAGIC %md ## Gold Layer Tests

# COMMAND ----------

print("=" * 70)
print("GOLD LAYER TESTS")
print("=" * 70)

gold = inventory_df.filter("layer = 'gold'").collect()

for test_type in ["unit", "integration"]:
    rows = [r for r in gold if r["test_type"] == test_type]
    if not rows:
        continue

    # Group by file
    files_seen = []
    file_groups = {}
    for r in rows:
        if r["file"] not in file_groups:
            file_groups[r["file"]] = []
            files_seen.append(r["file"])
        file_groups[r["file"]].append(r)

    print(f"\n{'─'*70}")
    print(f"  TYPE: {test_type.upper()}")
    print(f"{'─'*70}")

    for f in files_seen:
        file_rows = file_groups[f]
        print(f"\n  FILE:     {f}")
        print(f"  NOTEBOOK: {file_rows[0]['run_notebook']}")
        print(f"  TABLE:    {file_rows[0]['table_used']}")
        print(f"  TESTS:    {len(file_rows)}")
        for i, r in enumerate(file_rows, 1):
            print(f"\n    {i}. [{r['class']}]")
            print(f"       {r['test_name']}")
            print(f"       → {r['what_it_does']}")

# COMMAND ----------

# MAGIC %md ## Full Inventory Table

# COMMAND ----------

# Display the full inventory as a Databricks table
# Sort by layer order (bronze → silver → gold) then type (unit → integration)
from pyspark.sql import functions as F

display(
    inventory_df
    .withColumn("layer_order",
        F.when(F.col("layer") == "bronze", 1)
         .when(F.col("layer") == "silver", 2)
         .otherwise(3))
    .withColumn("type_order",
        F.when(F.col("test_type") == "unit", 1).otherwise(2))
    .orderBy("layer_order", "type_order", "file", "class", "test_name")
    .drop("layer_order", "type_order")
)

# COMMAND ----------

# MAGIC %md ## Summary counts

# COMMAND ----------

print("TEST COUNT SUMMARY")
print("=" * 50)
print(f"{'Layer':<10} {'Type':<15} {'Count':>6}")
print("-" * 50)

for layer in ["bronze", "silver", "gold"]:
    for ttype in ["unit", "integration"]:
        cnt = inventory_df.filter(
            (F.col("layer") == layer) & (F.col("test_type") == ttype)
        ).count()
        if cnt > 0:
            print(f"{layer:<10} {ttype:<15} {cnt:>6}")

total = inventory_df.count()
unit  = inventory_df.filter("test_type = 'unit'").count()
intg  = inventory_df.filter("test_type = 'integration'").count()

print("-" * 50)
print(f"{'Unit total':<25} {unit:>6}")
print(f"{'Integration total':<25} {intg:>6}")
print(f"{'GRAND TOTAL':<25} {total:>6}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where each test runs
# MAGIC
# MAGIC | Notebook | Runs | When |
# MAGIC |---|---|---|
# MAGIC | `run_tests_on_databricks.py` | All unit tests (tests/unit/) | **Before** run_pipeline.py |
# MAGIC | `run_tests_bronze.py` | Bronze integration tests | **After** run_pipeline.py |
# MAGIC | `run_tests_silver.py` | Silver integration tests | **After** run_pipeline.py |
# MAGIC | `run_tests_gold.py` | Gold integration tests | **After** run_pipeline.py |
