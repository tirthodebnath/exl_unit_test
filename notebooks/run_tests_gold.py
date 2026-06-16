# Databricks notebook source
"""
run_tests_gold.py
=================
PURPOSE:
    Run Gold layer data quality tests on the real gold_ogom_charges table.

WHEN TO RUN:
    After run_pipeline.py has loaded data into gold_ogom_charges.

TEST FILE:
    tests/integration/gold/test_ogom_charges.py (6 tests)

TESTS:
    1. test_no_new_charge_ids_in_gold
       Every charge_id in Gold must exist in Silver — no phantom charges.

    2. test_no_silver_charge_ids_dropped
       Every Silver charge_id must appear in Gold — LEFT JOIN guarantee.

    3. test_ogom_transaction_type_always_charge
       ogom_transaction_type must be 'Charge' on every row.

    4. test_charge_age_null_when_no_discharge
       charge_age must be null when discharge_date is null.

    5. test_discharge_date_after_admit_date
       Where both dates exist, discharge must be after admission.

    6. test_no_duplicate_join_combinations
       No (rcm_client_id + rcm_npi + patient_account_number)
       combination appears more than once in joined rows.

TABLES CHECKED:
    workspace.tirtho_db.gold_ogom_charges (primary)
    workspace.tirtho_db.silver_charges    (for tests 1, 2)

AUDIT:
    Results written to workspace.tirtho_db.test_audit_log
    with layer = 'gold' for all rows from this run.
"""

# COMMAND ----------
# MAGIC %pip install pytest==8.3.3 chispa==0.10.1 --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os, sys, shutil, uuid
from datetime import datetime
from pyspark.sql import SparkSession

NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils()
           .notebook().getContext().notebookPath().get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]
WORK_DIR  = "/tmp/exl_gold_tests"

if os.path.exists(WORK_DIR):
    try:
        shutil.rmtree(WORK_DIR)
    except (PermissionError, OSError):
        import random
        WORK_DIR = f"/tmp/exl_gold_tests_{random.randint(10000, 99999)}"

shutil.copytree(REPO_PATH, WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
sys.dont_write_bytecode = True

spark = SparkSession.builder.getOrCreate()

run_id        = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
workspace_url = "https://dbc-252798a7-44b9.cloud.databricks.com"
notebook_link = f"{workspace_url}/#notebook{NOTEBOOK_PATH}"

os.environ["AUDIT_RUN_ID"]        = run_id
os.environ["AUDIT_RUN_TIMESTAMP"] = run_timestamp
os.environ["AUDIT_NOTEBOOK_LINK"] = notebook_link
os.environ["AUDIT_TABLE"]         = "workspace.tirtho_db.test_audit_log"

print(f"Run ID:   {run_id}")
print(f"Notebook: {notebook_link}")

# ---------------------------------------------------------------------------
# Pre-flight checks — both required tables must exist before running tests
# ---------------------------------------------------------------------------
for t in [
    "workspace.tirtho_db.gold_ogom_charges",    # primary Gold table tested
    "workspace.tirtho_db.silver_charges",        # used for tests 1 and 2
]:
    try:
        spark.table(t).limit(1).count()
        print(f"✓ {t}")
    except Exception:
        raise RuntimeError(
            f"Table '{t}' not found. Run run_pipeline.py first."
        )

# COMMAND ----------
import pytest

print("=" * 60)
print("RUNNING GOLD DATA QUALITY TESTS")
print("=" * 60)
print()
print("File: tests/integration/gold/test_ogom_charges.py")
print("Tests: 6")
print()
print("  1. test_no_new_charge_ids_in_gold")
print("  2. test_no_silver_charge_ids_dropped")
print("  3. test_ogom_transaction_type_always_charge")
print("  4. test_charge_age_null_when_no_discharge")
print("  5. test_discharge_date_after_admit_date")
print("  6. test_no_duplicate_join_combinations")
print()

exit_code = pytest.main([
    "tests/integration/gold/test_ogom_charges.py",  # only this file
    "-v", "-ra", "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print(f"\npytest exit code: {exit_code}")

# COMMAND ----------
# Show this run's results from audit table
try:
    display(
        spark.table("workspace.tirtho_db.test_audit_log")
             .filter(f"run_id = '{run_id}'")
             .orderBy("test_name")
             .select("layer", "table_name", "test_name",
                     "status", "duration_seconds", "fail_message")
    )
except Exception:
    pass

assert exit_code == 0, (
    f"Gold tests FAILED (exit code {exit_code}). "
    f"See test_audit_log where run_id = '{run_id}'"
)
print("ALL GOLD TESTS PASSED.")
