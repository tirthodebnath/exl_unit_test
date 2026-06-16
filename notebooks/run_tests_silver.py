# Databricks notebook source
"""
run_tests_silver.py
===================
PURPOSE:
    Run Silver layer data quality tests on REAL Delta tables.

WHEN TO RUN:
    After run_pipeline.py has loaded data into silver_charges
    and silver_patientvisits Delta tables.

TESTS RUN (3 tests on silver_charges):
    test_service_date_is_datetime       — service_date is DateType
    test_post_date_not_before_service   — posting date never before service date
    test_no_exact_duplicate_rows        — charge_id is unique in Silver

AUDIT:
    Results written to workspace.tirtho_db.test_audit_log
    with layer = "silver" for all rows from this run.
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
WORK_DIR  = "/tmp/exl_silver_tests"

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

for t in ["workspace.tirtho_db.silver_charges",
          "workspace.tirtho_db.silver_patientvisits"]:
    try:
        spark.table(t).limit(1).count()
        print(f"✓ {t}")
    except Exception:
        raise RuntimeError(f"Table '{t}' not found. Run run_pipeline.py first.")

# COMMAND ----------
import pytest

print("=" * 60)
print("RUNNING SILVER DATA QUALITY TESTS")
print("=" * 60)

exit_code = pytest.main([
    "tests/integration/test_silver_realdata.py",
    "-v", "-ra", "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print(f"\npytest exit code: {exit_code}")

# COMMAND ----------
try:
    display(
        spark.table("workspace.tirtho_db.test_audit_log")
             .filter(f"run_id = '{run_id}'")
             .orderBy("test_name")
    )
except Exception:
    pass

assert exit_code == 0, (
    f"Silver tests FAILED (exit code {exit_code}). "
    f"See test_audit_log where run_id = '{run_id}'"
)
print("SILVER TESTS PASSED.")
