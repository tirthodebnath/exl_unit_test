# Databricks notebook source
"""
run_tests_gold.py
=================
PURPOSE:
    Run Gold layer data quality tests on REAL Delta tables.

WHEN TO RUN:
    After run_pipeline.py has loaded data into:
        gold_rcm_summary     (V1 — rcm_client_id join only)
        gold_rcm_summary_v2  (V2 — 3-condition + discharge filter join)

TESTS RUN (from tests/integration/gold/):
    test_charges.py (6 tests — charge-focused):
        test_charge_count_matches_silver
        test_total_charge_amount_matches_silver
        test_no_new_charge_ids_in_gold
        test_no_silver_charge_ids_dropped
        test_amount_band_valid_in_gold
        test_service_date_is_date_type_in_gold

    test_patientvisits.py (6 tests — visit-focused):
        test_discharge_date_not_null_where_visit_joined
        test_npi_matches_between_charge_and_visit
        test_patient_account_number_matches
        test_visit_columns_prefixed_pv
        test_left_join_keeps_all_charges
        test_rcm_client_id_not_null_in_gold

AUDIT:
    Results written to workspace.tirtho_db.test_audit_log
    with layer = "gold" for all rows from this run.
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
    shutil.rmtree(WORK_DIR)
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

# Check all required Gold tables exist
for t in ["workspace.tirtho_db.silver_charges",
          "workspace.tirtho_db.gold_rcm_summary",
          "workspace.tirtho_db.gold_rcm_summary_v2"]:
    try:
        spark.table(t).limit(1).count()
        print(f"✓ {t}")
    except Exception:
        raise RuntimeError(f"Table '{t}' not found. Run run_pipeline.py first.")

# COMMAND ----------
import pytest

print("=" * 60)
print("RUNNING GOLD DATA QUALITY TESTS")
print("=" * 60)
print("  test_charges.py      — charge reconciliation (6 tests)")
print("  test_patientvisits.py — visit join correctness (6 tests)")
print()

exit_code = pytest.main([
    "tests/integration/gold",   # runs both test_charges.py and test_patientvisits.py
    "-v", "-ra", "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print(f"\npytest exit code: {exit_code}")

# COMMAND ----------
# Show results split by test file (charges vs patientvisits)
try:
    audit_df = (
        spark.table("workspace.tirtho_db.test_audit_log")
             .filter(f"run_id = '{run_id}'")
             .orderBy("table_name", "test_name")
    )
    print(f"\nGold test results for run: {run_id}")
    display(audit_df.select(
        "table_name", "test_class", "test_name", "status",
        "duration_seconds", "fail_message"
    ))
except Exception:
    pass

assert exit_code == 0, (
    f"Gold tests FAILED (exit code {exit_code}). "
    f"See test_audit_log where run_id = '{run_id}'"
)
print("GOLD TESTS PASSED.")
