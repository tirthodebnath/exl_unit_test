"""
tests/integration/conftest.py
==============================
PURPOSE:
    Shared fixtures for integration tests.

    Unlike the unit test conftest (which creates fake DataFrames in memory),
    this conftest provides fixtures that read from REAL Delta tables. Every
    fixture here returns a DataFrame backed by an actual Delta table that
    was populated by run_pipeline.py.

IMPORTANT:
    These fixtures assume run_pipeline.py has already been executed and
    all 5 Delta tables exist in workspace.tirtho_db. If any table is missing,
    the fixture will raise a clear error explaining which table to create.

SPARK SESSION:
    Uses getOrCreate() — on Databricks a SparkSession is already running.
    We reuse it rather than creating a new one, which would be wasteful
    and could conflict with the existing session's configurations.
"""

import pytest
from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# SparkSession fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark():
    """
    Return the active SparkSession on Databricks.

    WHY scope="session":
        A SparkSession is expensive to create. Session scope means it is
        created once and shared across all integration tests in the run.
        This is safe because we never mutate the session's config in tests.

    Returns:
        SparkSession: The active (or newly created) Spark session.
    """
    return SparkSession.builder.getOrCreate()


# ---------------------------------------------------------------------------
# Real table fixtures — each reads from a live Delta table
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bronze_charges_real(spark):
    """
    Return the real bronze_charges Delta table as a DataFrame.

    WHY scope="session":
        The Bronze table does not change during the test run. Loading it
        once and reusing it avoids re-reading the Delta log on every test.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.bronze_charges.
    """
    try:
        return spark.table("workspace.tirtho_db.bronze_charges")
    except Exception as e:
        pytest.skip(
            f"bronze_charges table not found. Run run_pipeline.py first. "
            f"Error: {e}"
        )


@pytest.fixture(scope="session")
def bronze_patientvisits_real(spark):
    """
    Return the real bronze_patientvisits Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.bronze_patientvisits.
    """
    try:
        return spark.table("workspace.tirtho_db.bronze_patientvisits")
    except Exception as e:
        pytest.skip(
            f"bronze_patientvisits table not found. Run run_pipeline.py first. "
            f"Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_charges_real(spark):
    """
    Return the real silver_charges Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.silver_charges.
    """
    try:
        return spark.table("workspace.tirtho_db.silver_charges")
    except Exception as e:
        pytest.skip(
            f"silver_charges table not found. Run run_pipeline.py first. "
            f"Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_patientvisits_real(spark):
    """
    Return the real silver_patientvisits Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.silver_patientvisits.
    """
    try:
        return spark.table("workspace.tirtho_db.silver_patientvisits")
    except Exception as e:
        pytest.skip(
            f"silver_patientvisits table not found. Run run_pipeline.py first. "
            f"Error: {e}"
        )


@pytest.fixture(scope="session")
def gold_real(spark):
    """
    Return the real gold_rcm_summary Delta table as a DataFrame.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.gold_rcm_summary.
    """
    try:
        return spark.table("workspace.tirtho_db.gold_rcm_summary")
    except Exception as e:
        pytest.skip(
            f"gold_rcm_summary table not found. Run run_pipeline.py first. "
            f"Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_and_gold_real(silver_charges_real, gold_real):
    """
    Convenience fixture bundling Silver charges and Gold together.

    WHY:
        All Gold reconciliation tests need both Silver charges and Gold.
        This fixture avoids repeating the same two arguments in every
        Gold test method signature.

    Returns:
        tuple: (silver_charges DataFrame, gold_rcm_summary DataFrame)
    """
    return silver_charges_real, gold_real
