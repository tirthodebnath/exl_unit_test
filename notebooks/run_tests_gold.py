# Databricks notebook source
"""
run_tests_gold.py
=================
PURPOSE:
    Run Gold layer data quality tests on real Delta tables.

WHEN TO RUN:
    After run_pipeline.py has loaded data into gold_ogom_charges.

TESTS (6 tests on gold_ogom_charges):
    1. test_no_new_charge_ids_in_gold         — no phantom charge_ids
    2. test_no_silver_charge_ids_dropped      — no charges lost (LEFT JOIN)
    3. test_ogom_transaction_type_always_charge — always "Charge"
    4. test_charge_age_null_when_no_discharge  — null when no discharge
    5. test_discharge_date_after_admit_date    — discharge after admit
    6. test_no_duplicate_join_combinations     — join is 1-to-1

AUDIT:
    Results written directly to workspace.tirtho_db.test_audit_log
    from this notebook — not from pytest hooks.
"""

# COMMAND ----------
# MAGIC %md ## Step 1 — Install dependencies

# COMMAND ----------
# MAGIC %pip install pytest==8.3.3 --quiet

# COMMAND ----------
# MAGIC %md ## Step 2 — Restart Python

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Step 3 — Setup repo and audit environment

# COMMAND ----------
import os, sys, shutil, uuid
from datetime import datetime
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Locate repo and copy to writable path
# WHY: Workspace is read-only. pytest needs to write cache files to /tmp/
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# SparkSession — reuse existing Databricks session
# ---------------------------------------------------------------------------
spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Audit variables — generated once and used throughout this notebook
# ---------------------------------------------------------------------------
run_id        = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
workspace_url = "https://dbc-252798a7-44b9.cloud.databricks.com"
notebook_link = f"{workspace_url}/#notebook{NOTEBOOK_PATH}"

print(f"Run ID:        {run_id}")
print(f"Run timestamp: {run_timestamp}")
print(f"Notebook link: {notebook_link}")

# COMMAND ----------
# MAGIC %md ## Step 4 — Pre-flight checks

# COMMAND ----------
# ---------------------------------------------------------------------------
# Verify required tables exist before running tests
# ---------------------------------------------------------------------------
required = [
    "workspace.tirtho_db.gold_ogom_charges",
    "workspace.tirtho_db.silver_charges",
]
for t in required:
    try:
        spark.table(t).limit(1).count()
        print(f"✓ {t}")
    except Exception:
        raise RuntimeError(
            f"Table '{t}' not found. Run run_pipeline.py first."
        )

# COMMAND ----------
# MAGIC %md ## Step 5 — Run Gold data quality tests

# COMMAND ----------
import pytest
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# ---------------------------------------------------------------------------
# ResultCollector — inline pytest plugin that captures test results
# in the same notebook process.
#
# WHY NOT CONFTEST HOOKS:
#     After restartPython(), env vars and SparkSession may not be
#     available inside pytest hook subprocesses. This collector runs
#     inside this notebook process where SparkSession and run_id
#     are guaranteed to be set correctly.
# ---------------------------------------------------------------------------
class ResultCollector:
    """Captures every test result as pytest runs."""

    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        """
        Fires after each test phase.
        We only capture 'call' — the actual test execution.
        """
        if report.when != "call":
            return

        # Gold tests are plain functions — node ID has only 2 parts:
        # tests/integration/gold/test_ogom_charges.py::test_name
        test_name = report.nodeid.split("::")[-1]

        self.results.append({
            "run_id":           run_id,
            "run_timestamp":    run_timestamp,
            "layer":            "gold",
            "table_name":       "gold_ogom_charges",
            "test_class":       "none",
            "test_name":        test_name,
            "status":           "PASS" if report.passed else "FAIL",
            "fail_message":     str(report.longrepr)[-2000:]
                                if report.failed and report.longrepr
                                else None,
            "notebook_link":    notebook_link,
            "duration_seconds": round(report.duration, 3),
        })


print("=" * 60)
print("RUNNING GOLD DATA QUALITY TESTS")
print("=" * 60)
print()
print("  1. test_no_new_charge_ids_in_gold")
print("  2. test_no_silver_charge_ids_dropped")
print("  3. test_ogom_transaction_type_always_charge")
print("  4. test_charge_age_null_when_no_discharge")
print("  5. test_discharge_date_after_admit_date")
print("  6. test_no_duplicate_join_combinations")
print()

collector = ResultCollector()

exit_code = pytest.main([
    "tests/integration/gold/test_ogom_charges.py",
    "-v", "-ra", "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
], plugins=[collector])

print(f"\npytest exit code: {exit_code}")
print(f"Results collected: {len(collector.results)}")

# COMMAND ----------
# MAGIC %md ## Step 6 — Write results to audit table and display

# COMMAND ----------
# ---------------------------------------------------------------------------
# Write results directly from this notebook to the audit Delta table.
# ---------------------------------------------------------------------------
if collector.results:
    schema = StructType([
        StructField("run_id",           StringType(), True),
        StructField("run_timestamp",    StringType(), True),
        StructField("layer",            StringType(), True),
        StructField("table_name",       StringType(), True),
        StructField("test_class",       StringType(), True),
        StructField("test_name",        StringType(), True),
        StructField("status",           StringType(), True),
        StructField("fail_message",     StringType(), True),
        StructField("notebook_link",    StringType(), True),
        StructField("duration_seconds", FloatType(),  True),
    ])

    rows = [
        (
            r["run_id"], r["run_timestamp"], r["layer"],
            r["table_name"], r["test_class"], r["test_name"],
            r["status"], r["fail_message"], r["notebook_link"],
            r["duration_seconds"],
        )
        for r in collector.results
    ]

    df = spark.createDataFrame(rows, schema=schema)
    df.write.format("delta").mode("append") \
       .saveAsTable("workspace.tirtho_db.test_audit_log")

    print(f"✓ {len(rows)} results written to test_audit_log")
    print(f"  run_id: {run_id}")
    print()

    display(df.select(
        "layer", "table_name", "test_name",
        "status", "duration_seconds", "fail_message"
    ))
else:
    print("No results collected — pytest may have failed during collection")

# ---------------------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------------------
assert exit_code == 0, (
    f"Gold tests FAILED (exit code {exit_code}). "
    f"See test_audit_log where run_id = '{run_id}'"
)
print("ALL GOLD TESTS PASSED.")
