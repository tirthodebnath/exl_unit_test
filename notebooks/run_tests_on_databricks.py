# Databricks notebook source
# MAGIC %pip install pytest==8.3.3 chispa==0.10.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, shutil

NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils()
           .notebook().getContext().notebookPath().get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]
WORK_DIR  = "/tmp/exl-unit-test"
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(REPO_PATH, WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
sys.dont_write_bytecode = True
print(f"Ready: {WORK_DIR}")

# COMMAND ----------

import pytest
exit_code = pytest.main([
    "tests", "-ra", "--tb=short",
    "--override-ini=cache_dir=/tmp/.pytest_cache",
    "--basetemp=/tmp/pytest-temp",
])
print(f"\npytest exit code: {exit_code}")

# COMMAND ----------

assert exit_code == 0, f"Tests failed (exit code {exit_code})"
