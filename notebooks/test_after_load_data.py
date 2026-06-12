# Databricks notebook source
"""
test_after_load_data.py
=======================
PURPOSE:
    Run integration tests on REAL DATA after run_pipeline.py has loaded
    real data into Delta tables.

    Uses the EXACT SAME test logic as unit tests but with real Delta tables
    instead of dummy data. Results of every test are automatically written
    to workspace.tirtho_db.test_audit_log after the run completes.

AUDIT TABLE:
    Every test result (PASS/FAIL/SKIP) is automatically written to
    workspace.tirtho_db.test_audit_log by a pytest hook in conftest.py.
    When a new test is added to tests/integration/, it appears in the
    audit table on the next run — no manual changes needed.

WHEN TO RUN:
    STEP 3 — After run_pipeline.py has completed.
    Sequence:
        1. run_tests_on_databricks.py  (unit tests, dummy data, before pipeline)
        2. run_pipeline.py             (load real data into Delta tables)
        3. THIS NOTEBOOK               (integration tests + audit logging)

PRE-REQUISITES:
    Run these SQL scripts ONCE before using this notebook:
        notebooks/create_metadata_table.sql  — creates test_table_metadata
        notebooks/create_audit_table.sql     — creates test_audit_log
"""

# COMMAND ----------

# MAGIC %md ## Step 1 — Install test dependencies

# COMMAND ----------

# MAGIC %pip install pytest==8.3.3 chispa==0.10.1 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 2 — Set up working directory

# COMMAND ----------

import os, sys, shutil, uuid
from datetime import datetime
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Locate repo and copy to writable path
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

# COMMAND ----------

# MAGIC %md ## Step 3 — Set audit environment variables

# COMMAND ----------

# ---------------------------------------------------------------------------
# These environment variables are read by the pytest hooks in
# tests/integration/conftest.py. They are set here (in the notebook)
# so the hooks know where to write and how to identify this run.
#
# AUDIT_RUN_ID:
#     Unique ID per notebook execution — groups all 48 test rows together
#     so you can query "show me all results from the 10:30 run"
#
# AUDIT_RUN_TIMESTAMP:
#     Human-readable timestamp for the audit table — easier to read
#     in queries than a raw run_id
#
# AUDIT_NOTEBOOK_LINK:
#     Clickable URL stored in the audit table — lets you jump directly
#     to this notebook from a query result
#
# AUDIT_TABLE:
#     Fully qualified name of the audit Delta table
# ---------------------------------------------------------------------------

# Generate unique run ID: timestamp + short UUID
# Example: 20240527_103045_a3f7c2b1
run_id        = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Build the notebook URL — stored in each audit row for traceability
# Format: https://<workspace>/browse/notebooks/<path>
context       = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = "https://dbc-252798a7-44b9.cloud.databricks.com"
notebook_link = f"{workspace_url}/#notebook{NOTEBOOK_PATH}"

# Set environment variables — picked up by pytest hooks in conftest.py
os.environ["AUDIT_RUN_ID"]        = run_id
os.environ["AUDIT_RUN_TIMESTAMP"] = run_timestamp
os.environ["AUDIT_NOTEBOOK_LINK"] = notebook_link
os.environ["AUDIT_TABLE"]         = "workspace.tirtho_db.test_audit_log"

print(f"Run ID:        {run_id}")
print(f"Run timestamp: {run_timestamp}")
print(f"Notebook link: {notebook_link}")
print(f"Audit table:   workspace.tirtho_db.test_audit_log")

# COMMAND ----------

# MAGIC %md ## Step 4 — Pre-flight checks

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Check 1: Metadata table must exist
# test_after_load_data uses it to know which tables to test
# ---------------------------------------------------------------------------
METADATA_TABLE = "workspace.tirtho_db.test_table_metadata"
try:
    meta_count = spark.table(METADATA_TABLE).filter("active = true").count()
    print(f"✓ Metadata table OK — {meta_count} active table(s)")
except Exception as e:
    raise RuntimeError(
        f"Metadata table '{METADATA_TABLE}' not found. "
        f"Run notebooks/create_metadata_table.sql first. Error: {e}"
    )

# ---------------------------------------------------------------------------
# Check 2: Audit table must exist
# If it does not, the pytest hook will print a warning but not fail
# Better to catch this early with a clear message
# ---------------------------------------------------------------------------
AUDIT_TABLE = "workspace.tirtho_db.test_audit_log"
try:
    spark.table(AUDIT_TABLE).limit(1).count()
    print(f"✓ Audit table OK — results will be written here")
except Exception as e:
    print(
        f"⚠ Audit table '{AUDIT_TABLE}' not found. "
        f"Test results will NOT be logged. "
        f"Run notebooks/create_audit_table.sql to enable audit logging."
    )

# ---------------------------------------------------------------------------
# Check 3: All 5 pipeline Delta tables must exist
# If any are missing, tests will skip with unclear errors
# Better to catch this here with a specific message
# ---------------------------------------------------------------------------
required_tables = [
    "workspace.tirtho_db.bronze_charges",
    "workspace.tirtho_db.bronze_patientvisits",
    "workspace.tirtho_db.silver_charges",
    "workspace.tirtho_db.silver_patientvisits",
    "workspace.tirtho_db.gold_rcm_summary",
]
missing = []
for t in required_tables:
    try:
        spark.table(t).limit(1).count()
        print(f"✓ {t}")
    except Exception:
        missing.append(t)
        print(f"✗ {t} — NOT FOUND")

if missing:
    raise RuntimeError(
        f"These tables are missing: {missing}. "
        f"Run run_pipeline.py before this notebook."
    )

print(f"\nAll pre-flight checks passed. Starting integration tests...")

# COMMAND ----------

# MAGIC %md ## Step 5 — Run integration tests on real data

# COMMAND ----------

import pytest

print("=" * 60)
print("RUNNING INTEGRATION TESTS — REAL DELTA TABLE DATA")
print("=" * 60)
print(f"Run ID:  {run_id}")
print(f"Results: workspace.tirtho_db.test_audit_log")
print()

# ---------------------------------------------------------------------------
# Run pytest on tests/integration/ ONLY.
# The pytest hooks in tests/integration/conftest.py automatically:
#   1. Capture every test result as it completes
#   2. Write all results to test_audit_log after the session finishes
# No manual logging needed — any new test added to tests/integration/
# is automatically included in the audit table on the next run.
# ---------------------------------------------------------------------------
exit_code = pytest.main([
    "tests/integration",                            # integration tests ONLY
    "-v",                                           # verbose — show each test
    "-ra",                                          # show all non-passing
    "--tb=short",                                   # readable tracebacks
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print(f"\npytest exit code: {exit_code}")

# COMMAND ----------

# MAGIC %md ## Step 6 — Query audit table for this run's results

# COMMAND ----------

# ---------------------------------------------------------------------------
# Show the results from THIS run directly in the notebook output.
# The full history is in the audit table for querying later.
# ---------------------------------------------------------------------------
try:
    audit_df = (
        spark.table("workspace.tirtho_db.test_audit_log")
             .filter(f"run_id = '{run_id}'")
             .orderBy("layer", "table_name", "test_name")
    )

    total   = audit_df.count()
    passed  = audit_df.filter("status = 'PASS'").count()
    failed  = audit_df.filter("status = 'FAIL'").count()
    skipped = audit_df.filter("status = 'SKIP'").count()

    print(f"\nAudit log for run: {run_id}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print()

    # Show failed tests with their error messages
    if failed > 0:
        print("FAILED TESTS:")
        failed_rows = (
            audit_df.filter("status = 'FAIL'")
                    .select("layer", "table_name", "test_name", "fail_message")
                    .collect()
        )
        for row in failed_rows:
            print(f"\n  ✗ [{row['layer']}] {row['table_name']} → {row['test_name']}")
            print(f"    {row['fail_message'][:300] if row['fail_message'] else 'No message'}")

    display(audit_df.select(
        "layer", "table_name", "test_class", "test_name",
        "status", "duration_seconds", "fail_message"
    ))

except Exception as e:
    print(f"Could not query audit table: {e}")

# COMMAND ----------

# MAGIC %md ## Step 7 — Final gate

# COMMAND ----------

# ---------------------------------------------------------------------------
# Raise if any test failed.
# This marks the Databricks job as FAILED so it is visible in job history
# and triggers any failure alerts configured on the job.
# ---------------------------------------------------------------------------
assert exit_code == 0, (
    f"Integration tests FAILED on real data (exit code {exit_code}). "
    f"Check workspace.tirtho_db.test_audit_log for run_id='{run_id}' "
    f"to see which tests failed and why."
)

print("ALL INTEGRATION TESTS PASSED.")
print(f"Full results in: workspace.tirtho_db.test_audit_log")
print(f"Filter by run_id = '{run_id}' to see this run's results.")
