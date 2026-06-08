# Databricks notebook source
"""
test_after_load_data.py
=======================
PURPOSE:
    Run the INTEGRATION TEST suite on REAL DATA after run_pipeline.py
    has loaded real data into the Delta tables.

    Uses the EXACT SAME test logic as the unit tests but with real tables
    instead of dummy data:
        Unit test                     → Integration test equivalent
        make_charge() dummy rows      → spark.table("bronze_charges")
        hardcoded TABLE_METADATA list → workspace.tirtho_db.test_table_metadata
        CSV files in /tmp             → /Volumes/.../charges.csv

WHEN TO RUN:
    STEP 3 — Run this AFTER run_pipeline.py has completed.
    Sequence:
        1. run_tests_on_databricks.py  (unit tests, dummy data)
        2. run_pipeline.py             (load real data into Delta tables)
        3. THIS NOTEBOOK               (integration tests, real data)

WHAT IT RUNS:
    tests/integration/ ONLY:
        - test_bronze_realdata.py:  17 Bronze tests × 2 tables = 34 tests
        - test_silver_realdata.py:  3 Silver tests
        - test_gold_realdata.py:    11 Gold tests
    Total: 48 tests on real Delta table data.

TABLE_METADATA SOURCE:
    Read from workspace.tirtho_db.test_table_metadata (the Delta metadata table).
    Run notebooks/create_metadata_table.sql once before using this notebook.

OUTPUT:
    Exit code 0 = all checks passed on real data
    Exit code 1 = real data has quality issues — investigate before downstream use
"""

# COMMAND ----------

# MAGIC %md ## Step 1 — Install test dependencies

# COMMAND ----------

# MAGIC %pip install pytest==8.3.3 chispa==0.10.1 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 2 — Set up working directory and imports

# COMMAND ----------

import os, sys, shutil

# ---------------------------------------------------------------------------
# Locate repo root and copy to a writable path
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = (
    dbutils.notebook.entry_point
           .getDbutils().notebook().getContext().notebookPath().get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]
WORK_DIR  = "/tmp/exl_after_load_work"

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(REPO_PATH, WORK_DIR)

sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
sys.dont_write_bytecode = True

print(f"Repo source: {REPO_PATH}")
print(f"Work dir:    {WORK_DIR}")

# ---------------------------------------------------------------------------
# Verify the metadata table exists before running 48 tests
# If it does not exist, stop here with a clear setup instruction
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

METADATA_TABLE = "workspace.tirtho_db.test_table_metadata"
try:
    meta_count = spark.table(METADATA_TABLE).filter("active = true").count()
    print(f"Metadata table OK: {meta_count} active table(s) found")
except Exception as e:
    raise RuntimeError(
        f"Metadata table '{METADATA_TABLE}' not found or not readable. "
        f"Run notebooks/create_metadata_table.sql before this notebook. "
        f"Error: {e}"
    )

# ---------------------------------------------------------------------------
# Verify all 5 Delta tables exist before running tests
# If any are missing, the test run would fail with confusing errors
# ---------------------------------------------------------------------------
required_tables = [
    "workspace.tirtho_db.bronze_charges",
    "workspace.tirtho_db.bronze_patientvisits",
    "workspace.tirtho_db.silver_charges",
    "workspace.tirtho_db.silver_patientvisits",
    "workspace.tirtho_db.gold_rcm_summary",
]
missing_tables = []
for t in required_tables:
    try:
        spark.table(t).limit(1).count()
        print(f"  ✓ {t}")
    except Exception:
        missing_tables.append(t)
        print(f"  ✗ {t} — NOT FOUND")

if missing_tables:
    raise RuntimeError(
        f"These tables are missing: {missing_tables}. "
        f"Run run_pipeline.py before running this notebook."
    )

print("\nAll required tables found — safe to run integration tests.")

# COMMAND ----------

# MAGIC %md ## Step 3 — Run integration tests on real data

# COMMAND ----------

import pytest

print("=" * 60)
print("RUNNING INTEGRATION TESTS — REAL DATA (tests/integration/ only)")
print("=" * 60)
print()
print("Same test logic as unit tests, but reading from real Delta tables.")
print("TABLE_METADATA loaded from:", METADATA_TABLE)
print()

exit_code = pytest.main([
    "tests/integration",                            # integration tests ONLY
    "-ra",                                          # show all non-passing
    "--tb=short",                                   # readable tracebacks
    "-v",                                           # verbose — show each test name
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print()
print(f"pytest exit code: {exit_code}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Final gate

# COMMAND ----------

assert exit_code == 0, (
    f"Integration tests FAILED on real data (exit code {exit_code}). "
    f"Review the test output above to see which checks failed and why. "
    f"Fix the data quality issues before using Gold for downstream reporting."
)

print("ALL INTEGRATION TESTS PASSED on real data.")
print("Gold output is verified and ready for downstream consumption.")
