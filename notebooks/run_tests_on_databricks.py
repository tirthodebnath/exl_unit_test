# Databricks notebook source
"""
run_tests_on_databricks.py
==========================
PURPOSE:
    Run the UNIT TEST suite on DUMMY DATA before the real pipeline executes.
    This notebook proves that all transformation functions are written correctly
    before any real data is touched.

WHEN TO RUN:
    STEP 1 — Always run this FIRST.
    If any test fails here, fix the code before running run_pipeline.py.
    Never load real data with broken transformation code.

WHAT IT RUNS:
    tests/unit/ ONLY — dummy data, no Delta tables required.
    - test_bronze.py:          17 Bronze scenarios × 2 tables = 34 tests
    - test_silver_charges.py:  3 Silver business rule tests
    - test_gold.py:            11 Gold join and accuracy tests
    Total: 48 tests on fake in-memory data.

WHAT IT DOES NOT RUN:
    tests/integration/ — those use real Delta tables and run after the pipeline.
    test_after_load_data.py is the notebook for real-data tests.

OUTPUT:
    Exit code 0 = all tests passed = safe to run run_pipeline.py
    Exit code 1 = tests failed     = fix code before loading real data
"""

# COMMAND ----------

# MAGIC %md ## Step 1 — Install test dependencies

# COMMAND ----------

# Install pytest and chispa — not pre-installed on Databricks serverless
# --quiet suppresses verbose output
# MAGIC %pip install pytest==8.3.3 chispa==0.10.1 --quiet

# COMMAND ----------

# Restart Python so newly installed libraries are importable
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 2 — Set up working directory

# COMMAND ----------

import os, sys, shutil

# ---------------------------------------------------------------------------
# Locate the repo root from this notebook's Workspace path
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = (
    dbutils.notebook.entry_point
           .getDbutils().notebook().getContext().notebookPath().get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]

# /tmp/exl_unit_test_work is a writable subdirectory — never use /tmp/ directly
# because shutil.rmtree('/tmp/') fails on Databricks with PermissionError
WORK_DIR = "/tmp/exl_unit_test_work"

# Clean up previous run and copy fresh repo
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(REPO_PATH, WORK_DIR)

sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
sys.dont_write_bytecode = True   # no __pycache__ in Workspace

print(f"Repo source: {REPO_PATH}")
print(f"Work dir:    {WORK_DIR}")
print(f"Working dir: {os.getcwd()}")

# Sanity check — confirm src is importable before running 48 tests
try:
    import src.common.schemas
    print(f"src OK: {src.common.schemas.__file__}")
except ImportError as e:
    raise RuntimeError(f"Cannot import src — check REPO_PATH. Error: {e}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Run unit tests on dummy data

# COMMAND ----------

import pytest

print("=" * 60)
print("RUNNING UNIT TESTS — DUMMY DATA (tests/unit/ only)")
print("=" * 60)
print()

exit_code = pytest.main([
    "tests/unit",                                   # unit tests ONLY
    "-ra",                                          # show all non-passing
    "--tb=short",                                   # readable tracebacks
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])

print()
print(f"pytest exit code: {exit_code}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Gate: block pipeline if tests failed

# COMMAND ----------

assert exit_code == 0, (
    f"Unit tests FAILED (exit code {exit_code}). "
    f"Fix failing tests before running run_pipeline.py."
)

print("ALL UNIT TESTS PASSED — safe to proceed to run_pipeline.py")
