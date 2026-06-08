# Databricks notebook source
"""
run_tests_on_databricks.py
==========================
PURPOSE:
    Run the full unit test suite on DUMMY DATA before the real pipeline
    executes. This notebook proves that all transformation functions are
    written correctly before any real data is touched.

WHEN TO RUN:
    STEP 1 — Always run this FIRST.
    If any test fails here, fix the code before running run_pipeline.py.
    Never load real data with broken transformation code.

WHAT IT TESTS:
    - Bronze: 17 ingestion scenarios (parametrized for charges + patientvisits)
    - Silver: 3 business rule and transformation tests for charges
    - Gold:   10 join correctness and data accuracy tests
    All 48 tests use dummy/fake data — no real CSV or Delta table is touched.

OUTPUT:
    pytest exit code 0 = all tests passed = safe to run pipeline
    pytest exit code 1 = tests failed = DO NOT run pipeline
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install test dependencies
# MAGIC
# MAGIC pytest and chispa are not pre-installed on Databricks serverless.
# MAGIC We install them here before restarting Python so they are available
# MAGIC when the test suite runs.

# COMMAND ----------

# Install testing libraries
# --quiet suppresses the verbose install output in notebook
# MAGIC %pip install pytest==8.3.3 chispa==0.10.1 --quiet

# COMMAND ----------

# Restart Python so the newly installed libraries are importable.
# Without this, the pip install above would not take effect.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Set up working directory
# MAGIC
# MAGIC The Workspace Git folder is READ-ONLY on Databricks serverless.
# MAGIC pytest needs to write __pycache__, .pytest_cache and temp files.
# MAGIC We copy the entire repo to /tmp/exl_unit_test_work (a writable path)
# MAGIC and run everything from there.

# COMMAND ----------

import os
import sys
import shutil

# ---------------------------------------------------------------------------
# Find the repo root from the current notebook's path.
# getDbutils().notebook().getContext().notebookPath() gives the full path
# of this notebook inside the Databricks Workspace.
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = (
    dbutils.notebook.entry_point
           .getDbutils()
           .notebook()
           .getContext()
           .notebookPath()
           .get()
)

# Repo root = everything before /notebooks/...
# Example: /Workspace/DataBricks_Notebooks/Unit_Test/exl_unit_test
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]

# Writable copy of the repo — pytest runs from here, not from Workspace
# Using a specific subdirectory name (not /tmp/ directly) to avoid
# PermissionError when rmtree tries to delete a system-protected path
WORK_DIR = "/tmp/exl_unit_test_work"

# ---------------------------------------------------------------------------
# Clean up any previous run's work directory so we always start fresh.
# Fresh start prevents stale .pyc files from a previous code version
# interfering with the current test run.
# ---------------------------------------------------------------------------
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

# Copy the entire repo into the writable work directory
shutil.copytree(REPO_PATH, WORK_DIR)

# Add work dir to Python path so `from src.xxx import yyy` works
sys.path.insert(0, WORK_DIR)

# Change working directory so pytest finds pytest.ini and the tests/ folder
os.chdir(WORK_DIR)

# Prevent Python from writing __pycache__ into the read-only Workspace
sys.dont_write_bytecode = True

print(f"Repo source:  {REPO_PATH}")
print(f"Work dir:     {WORK_DIR}")
print(f"Working dir:  {os.getcwd()}")

# ---------------------------------------------------------------------------
# Sanity check: confirm src package is importable before running tests.
# If this fails, the path setup above went wrong and all 48 tests would
# fail with ImportError — better to know now with a clear message.
# ---------------------------------------------------------------------------
try:
    import src.common.schemas
    print(f"src import OK: {src.common.schemas.__file__}")
except ImportError as e:
    raise RuntimeError(
        f"Cannot import src package from {WORK_DIR}. "
        f"Check that REPO_PATH is correct. Error: {e}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Run the full unit test suite

# COMMAND ----------

import pytest

print("=" * 60)
print("RUNNING UNIT TESTS ON DUMMY DATA")
print("=" * 60)
print("These tests do NOT touch real data.")
print("They prove the transformation functions are written correctly.")
print()

# ---------------------------------------------------------------------------
# Run pytest programmatically.
# Args explained:
#   "tests"                          — test root directory
#   "-ra"                            — show summary of all non-passing tests
#   "--tb=short"                     — short traceback (readable in notebook)
#   "--override-ini=cache_dir=..."   — write cache to writable /tmp path
#   "--basetemp=..."                 — write temp files to writable /tmp path
# ---------------------------------------------------------------------------
exit_code = pytest.main([
    "tests",
    "-ra",
    "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print()
print("=" * 60)
print(f"pytest exit code: {exit_code}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Gate: only proceed if all tests passed

# COMMAND ----------

# ---------------------------------------------------------------------------
# This assertion acts as a hard gate.
# If any test failed, this cell raises AssertionError and the Databricks
# job stops here — preventing a broken pipeline from running on real data.
# ---------------------------------------------------------------------------
assert exit_code == 0, (
    f"Unit tests FAILED (pytest exit code {exit_code}). "
    f"Fix the failing tests before running run_pipeline.py."
)

print("ALL TESTS PASSED — safe to run run_pipeline.py")
