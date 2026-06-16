"""
tests/integration/gold/conftest.py
====================================
PURPOSE:
    Fixtures specific to Gold layer integration tests.

    Provides DataFrames for both Gold tables:
        gold_real      — workspace.tirtho_db.gold_rcm_summary   (V1 join)
        gold_v2_real   — workspace.tirtho_db.gold_rcm_summary_v2 (V2 join)

    Also provides combined fixtures that bundle Silver charges with each
    Gold table — used by reconciliation tests that compare Silver to Gold.
"""
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Active SparkSession — reused across all Gold tests.

    WHY scope="session":
        SparkSession is expensive to create. Session scope means one
        session shared across all Gold test files in this folder.
    """
    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="session")
def silver_charges_real(spark):
    """
    Real silver_charges Delta table — used for Gold reconciliation checks.

    WHY NEEDED IN GOLD TESTS:
        Gold reconciliation tests compare Gold output against Silver charges.
        This fixture loads Silver charges once for the whole Gold test session.

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
def gold_real(spark):
    """
    Real gold_rcm_summary Delta table (V1 — rcm_client_id join only).

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
def gold_v2_real(spark):
    """
    Real gold_rcm_summary_v2 Delta table (V2 — 3-condition + discharge filter).

    This is the enhanced Gold join:
        charges.rcm_client_id = visits.rcm_client_id
        AND charges.rcm_npi = visits.rcm_npi
        AND charges.patient_account_number = visits.patient_account_number
        AND visits.patient_discharge_date IS NOT NULL

    Returns:
        DataFrame: All rows from workspace.tirtho_db.gold_rcm_summary_v2.
    """
    try:
        return spark.table("workspace.tirtho_db.gold_rcm_summary_v2")
    except Exception as e:
        pytest.skip(
            f"gold_rcm_summary_v2 not found. Run run_pipeline.py first. Error: {e}"
        )


@pytest.fixture(scope="session")
def silver_and_gold_v2(silver_charges_real, gold_v2_real):
    """
    Bundle Silver charges and Gold V2 together for reconciliation tests.

    WHY:
        All charge reconciliation tests need both Silver charges and Gold V2.
        One fixture avoids repeating two arguments in every test method.

    Returns:
        tuple: (silver_charges DataFrame, gold_rcm_summary_v2 DataFrame)
    """
    return silver_charges_real, gold_v2_real
