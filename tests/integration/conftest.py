"""
tests/integration/conftest.py
==============================
PURPOSE:
    Shared fixtures and pytest hooks for integration tests.

    Contains two sections:
    1. FIXTURES      — SparkSession and real Delta table fixtures
    2. PYTEST HOOKS  — automatically capture every test result and
                       write to the audit Delta table

AUDIT TABLE AUTO-UPDATE:
    The pytest_runtest_logreport hook fires after EVERY test that runs.
    This means:
    - When you add a new test to tests/integration/, it automatically
      appears in the audit table on the next run
    - No code changes needed — the hook captures everything
    - New test = new row in test_audit_log automatically

    The hook reads environment variables set by test_after_load_data.py:
        AUDIT_RUN_ID         unique ID for this run
        AUDIT_RUN_TIMESTAMP  when the run started
        AUDIT_NOTEBOOK_LINK  URL to the notebook
        AUDIT_TABLE          fully qualified audit Delta table name
"""

import os
import re
import pytest
from pyspark.sql import SparkSession


# ===========================================================================
# SECTION 1: SPARK SESSION AND TABLE FIXTURES
# ===========================================================================

@pytest.fixture(scope="session")
def spark():
    """
    Return the active SparkSession on Databricks.

    WHY scope="session":
        Created once, shared across all 48 integration tests.
        Avoids the overhead of creating a new session per test.

    WHY getOrCreate():
        On Databricks a session is already running. We reuse it
        rather than creating a new local session which cannot
        read Delta tables from Unity Catalog.

    Returns:
        SparkSession: Active Databricks Spark session.
    """
    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="session")
def bronze_charges_real(spark):
    """
    Real bronze_charges Delta table as a DataFrame.

    WHY scope="session":
        Table does not change during test run.
        Load once, reuse across all 17 Bronze charges tests.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.bronze_charges.
    """
    try:
        return spark.table("workspace.tirtho_db.bronze_charges")
    except Exception as e:
        pytest.skip(
            f"bronze_charges not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def bronze_patientvisits_real(spark):
    """
    Real bronze_patientvisits Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.bronze_patientvisits.
    """
    try:
        return spark.table("workspace.tirtho_db.bronze_patientvisits")
    except Exception as e:
        pytest.skip(
            f"bronze_patientvisits not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_charges_real(spark):
    """
    Real silver_charges Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.silver_charges.
    """
    try:
        return spark.table("workspace.tirtho_db.silver_charges")
    except Exception as e:
        pytest.skip(
            f"silver_charges not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_patientvisits_real(spark):
    """
    Real silver_patientvisits Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.silver_patientvisits.
    """
    try:
        return spark.table("workspace.tirtho_db.silver_patientvisits")
    except Exception as e:
        pytest.skip(
            f"silver_patientvisits not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def gold_real(spark):
    """
    Real gold_rcm_summary Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.gold_rcm_summary.
    """
    try:
        return spark.table("workspace.tirtho_db.gold_rcm_summary")
    except Exception as e:
        pytest.skip(
            f"gold_rcm_summary not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_and_gold_real(silver_charges_real, gold_real):
    """
    Bundles Silver charges and Gold together for Gold tests.

    WHY:
        All Gold reconciliation tests need both DataFrames.
        One fixture instead of two arguments in every test.

    Returns:
        tuple: (silver_charges DataFrame, gold_rcm_summary DataFrame)
    """
    return silver_charges_real, gold_real


# ===========================================================================
# SECTION 2: PYTEST HOOKS FOR AUDIT TABLE
# ===========================================================================

# ---------------------------------------------------------------------------
# Module-level list: collects all test results during the session.
# Written to Delta in one batch when the session finishes.
# Starts empty on every fresh pytest run.
# ---------------------------------------------------------------------------
_audit_results = []


def _parse_test_info(nodeid: str) -> dict:
    """
    Parse layer, table_name, test_class, test_name from a pytest node ID.

    WHY THIS EXISTS:
        pytest identifies each test by a node ID string. We need to extract
        structured fields from it to store in the audit table columns.

    NODE ID FORMATS — 3 patterns handled:

        Pattern 1: class-based parametrized (Bronze)
            tests/integration/test_bronze_realdata.py
            ::TestBronzeIngestion
            ::test_duplicate_check[charges]
            parts = [file, class, test[table]]

        Pattern 2: class-based not parametrized (Silver)
            tests/integration/test_silver_realdata.py
            ::TestSilverServiceDate
            ::test_service_date_is_datetime
            parts = [file, class, test]

        Pattern 3: plain function — no class (Gold ogom)
            tests/integration/gold/test_ogom_charges.py
            ::test_no_new_charge_ids_in_gold
            parts = [file, test]

    WHY PATTERN 3 WAS RETURNING unknown:
        Old code assumed parts[2] was the test name (class-based format).
        For plain functions, parts[1] IS the test name and parts[2] does
        not exist — so table_name and layer both fell back to "unknown".

    Args:
        nodeid (str): Full pytest node ID string.

    Returns:
        dict: layer, table_name, test_class, test_name.
    """
    # Split node ID into its path components
    parts = nodeid.split("::")

    # ── Determine layer from file name ───────────────────────────────────────
    # Extract just the filename (last segment before .py)
    file_name = parts[0].split("/")[-1].replace(".py", "")

    if "bronze" in file_name:
        layer = "bronze"
    elif "silver" in file_name:
        layer = "silver"
    elif "gold" in file_name:
        layer = "gold"
    else:
        layer = "unknown"

    # ── Detect which pattern we are dealing with ─────────────────────────────
    # Pattern 3: plain function — only 2 parts (file + test_name)
    # Pattern 1/2: class-based — 3 parts (file + class + test_name)
    if len(parts) == 2:
        # Pattern 3: tests/integration/gold/test_ogom_charges.py::test_name
        # parts[1] is the test name directly — no class wrapper
        test_class = "none"
        test_part  = parts[1]
    else:
        # Pattern 1/2: file::ClassName::test_name or file::ClassName::test_name[table]
        test_class = parts[1]
        test_part  = parts[2]

    # ── Extract test name and table_name from the last part ──────────────────
    if "[" in test_part:
        # Parametrized test — table name is inside the square brackets
        # e.g. test_duplicate_check[charges] → test_name=test_duplicate_check, table=charges
        test_name  = test_part.split("[")[0]
        table_name = test_part.split("[")[1].rstrip("]")
    else:
        # Not parametrized — derive table name from layer
        test_name = test_part
        if layer == "bronze":
            table_name = "bronze_charges + bronze_patientvisits"
        elif layer == "silver":
            table_name = "silver_charges"
        elif layer == "gold":
            # Gold tests run on gold_ogom_charges
            table_name = "gold_ogom_charges"
        else:
            table_name = "unknown"

    return {
        "layer":      layer,
        "table_name": table_name,
        "test_class": test_class,
        "test_name":  test_name,
    }


def pytest_runtest_logreport(report):
    """
    Pytest hook — fires after each phase of every test.

    WHY THIS HOOK:
        This is how pytest allows external systems to intercept test results.
        It fires three times per test (setup, call, teardown). We only care
        about the 'call' phase which is the actual test execution.

        By appending to _audit_results here, we collect results for every
        test automatically — including any new tests added in the future.
        No manual registration needed. Add a test → it appears in the
        audit table on the next run.

    Args:
        report: pytest's TestReport object containing all test result info.
    """
    # Only capture the 'call' phase — this is the actual test execution.
    # 'setup' and 'teardown' phases are infrastructure, not the test itself.
    if report.when != "call":
        return

    # Parse the structured fields from the pytest node ID
    info = _parse_test_info(report.nodeid)

    # ── Determine status and fail message ───────────────────────────────────
    if report.passed:
        status       = "PASS"
        fail_message = None   # null in Delta — test passed, nothing to report

    elif report.failed:
        status = "FAIL"
        # Extract the actual assertion error message.
        # report.longrepr contains the full traceback — we take the last
        # line which has the actual AssertionError message, truncated to
        # 2000 chars to avoid Delta row size issues.
        if report.longrepr:
            full_msg     = str(report.longrepr)
            fail_message = full_msg[-2000:] if len(full_msg) > 2000 else full_msg
        else:
            fail_message = "Test failed — no error message captured"

    else:
        # Skipped — e.g. Delta table not found, metadata table missing
        status = "SKIP"
        fail_message = str(report.longrepr)[:500] if report.longrepr else "Skipped"

    # ── Build the audit row ─────────────────────────────────────────────────
    _audit_results.append({
        "run_id":           os.environ.get("AUDIT_RUN_ID",         "unknown"),
        "run_timestamp":    os.environ.get("AUDIT_RUN_TIMESTAMP",  "unknown"),
        "layer":            info["layer"],
        "table_name":       info["table_name"],
        "test_class":       info["test_class"],
        "test_name":        info["test_name"],
        "status":           status,
        "fail_message":     fail_message,
        "notebook_link":    os.environ.get("AUDIT_NOTEBOOK_LINK",  "unknown"),
        "duration_seconds": round(report.duration, 3),
    })


def pytest_sessionfinish(session, exitstatus):
    """
    Pytest hook — fires once after ALL tests have completed.

    WHY HERE AND NOT IN pytest_runtest_logreport:
        Writing to Delta on every single test would mean 48 separate Delta
        write operations — very slow and creates 48 small files.
        By collecting results in _audit_results and writing once here,
        we do one efficient batch write at the end of the session.

    WHAT IT DOES:
        1. Checks if there are any results to write
        2. Gets SparkSession
        3. Creates a DataFrame from _audit_results
        4. Appends to the audit Delta table
        5. Prints confirmation with row count

    FAILURE HANDLING:
        If the write fails (e.g. audit table not created yet), it prints
        a warning but does NOT fail the test run. The test results are
        not lost — they are just not persisted to Delta.

    Args:
        session:    The pytest Session object.
        exitstatus: The overall exit code (0=passed, 1=failed, etc.)
    """
    # Nothing to write if no tests ran (e.g. collection error)
    if not _audit_results:
        print("\nNo test results to write to audit table.")
        return

    audit_table = os.environ.get(
        "AUDIT_TABLE",
        "workspace.tirtho_db.test_audit_log"
    )

    try:
        from pyspark.sql.types import (
            FloatType, StringType, StructField, StructType
        )

        spark = SparkSession.builder.getOrCreate()

        # ── Define schema explicitly ────────────────────────────────────────
        # Explicit schema prevents Spark from inferring types incorrectly
        # on edge cases like null fail_message or very short duration
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

        # ── Convert results list to tuple list for createDataFrame ──────────
        rows = [
            (
                r["run_id"],
                r["run_timestamp"],
                r["layer"],
                r["table_name"],
                r["test_class"],
                r["test_name"],
                r["status"],
                r["fail_message"],
                r["notebook_link"],
                r["duration_seconds"],
            )
            for r in _audit_results
        ]

        # ── Write to Delta audit table (append — never overwrite) ────────────
        # append mode preserves historical runs — you can track quality over time
        df = spark.createDataFrame(rows, schema=schema)
        df.write.format("delta").mode("append").saveAsTable(audit_table)

        # Count pass/fail for summary
        passed = sum(1 for r in _audit_results if r["status"] == "PASS")
        failed = sum(1 for r in _audit_results if r["status"] == "FAIL")
        skipped = sum(1 for r in _audit_results if r["status"] == "SKIP")

        print(f"\n{'=' * 60}")
        print(f"AUDIT LOG WRITTEN TO: {audit_table}")
        print(f"  Run ID:  {_audit_results[0]['run_id']}")
        print(f"  Total:   {len(rows)} tests")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {failed}")
        print(f"  Skipped: {skipped}")
        print(f"{'=' * 60}")

    except Exception as e:
        # Do not fail the test run if audit write fails
        # Test results are printed above in the pytest output — not lost
        print(f"\n{'!' * 60}")
        print(f"WARNING: Could not write to audit table '{audit_table}'.")
        print(f"Reason: {e}")
        print(f"Run notebooks/create_audit_table.sql first.")
        print(f"{'!' * 60}")
