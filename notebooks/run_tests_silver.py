# Databricks notebook source
"""
run_tests_silver.py
===================
PURPOSE:
    Run Silver layer data quality tests on real Delta tables.

WHEN TO RUN:
    After run_pipeline.py has loaded data into silver_charges
    and silver_patientvisits Delta tables.

TESTS (3 tests on silver_charges):
    1. test_service_date_is_datetime      — service_date is DateType
    2. test_post_date_not_before_service  — no posting before service date
    3. test_no_exact_duplicate_rows       — charge_id is unique

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
WORK_DIR  = "/tmp/exl_silver_tests"

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
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
    "workspace.tirtho_db.silver_charges",
    "workspace.tirtho_db.silver_patientvisits",
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
# MAGIC %md ## Step 5 — Run Silver data quality tests

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

        # Extract class and test name from node ID
        # e.g. test_silver_realdata.py::TestSilverServiceDate::test_service_date_is_datetime
        parts      = report.nodeid.split("::")
        test_class = parts[1] if len(parts) > 1 else "none"
        test_name  = parts[-1]

        self.results.append({
            "run_id":           run_id,
            "run_timestamp":    run_timestamp,
            "layer":            "silver",
            "table_name":       "silver_charges",
            "test_class":       test_class,
            "test_name":        test_name,
            "status":           "PASS" if report.passed else "FAIL",
            "fail_message":     str(report.longrepr)[-2000:]
                                if report.failed and report.longrepr
                                else None,
            "notebook_link":    notebook_link,
            "duration_seconds": round(report.duration, 3),
        })


print("=" * 60)
print("RUNNING SILVER DATA QUALITY TESTS")
print("=" * 60)
print()

collector = ResultCollector()

exit_code = pytest.main([
    "tests/integration/test_silver_realdata.py",
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
    f"Silver tests FAILED (exit code {exit_code}). "
    f"See test_audit_log where run_id = '{run_id}'"
)
print("ALL SILVER TESTS PASSED.")
