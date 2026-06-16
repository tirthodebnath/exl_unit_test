"""
src/pipeline.py
===============
PURPOSE:
    Orchestrates the full RCM pipeline — Bronze → Silver → Gold.
    run_pipeline() is a pure function (no file reads, no table writes).
    Used by both the pipeline notebook (real data) and unit tests (dummy data).

GOLD OUTPUTS:
    rcm_summary    — original join on rcm_client_id only
    rcm_summary_v2 — enhanced join (rcm_client_id + rcm_npi + patient_account_number
                     + patient_discharge_date IS NOT NULL)
"""
from typing import Dict
from pyspark.sql import DataFrame, SparkSession

from src.bronze.ingest import ingest_charges, ingest_patientvisits
from src.silver.transform_charges import build_silver_charges
from src.silver.transform_patientvisits import build_silver_patientvisits
from src.gold.aggregate import build_rcm_summary, build_rcm_summary_v2


def run_pipeline(
    bronze_charges: DataFrame,
    bronze_patientvisits: DataFrame,
) -> Dict[str, DataFrame]:
    """
    Pure function form of the full pipeline.

    Args:
        bronze_charges       (DataFrame): Raw charges from Bronze ingest.
        bronze_patientvisits (DataFrame): Raw visits from Bronze ingest.

    Returns:
        dict with keys:
            silver_charges       — cleaned charges
            silver_patientvisits — cleaned visits
            rcm_summary          — Gold V1 (rcm_client_id join only)
            rcm_summary_v2       — Gold V2 (3-condition + discharge filter join)
    """
    # Silver — clean and enrich both tables
    silver_charges       = build_silver_charges(bronze_charges)
    silver_patientvisits = build_silver_patientvisits(bronze_patientvisits)

    return {
        "silver_charges":       silver_charges,
        "silver_patientvisits": silver_patientvisits,
        # Original Gold join — rcm_client_id only, one-to-many
        "rcm_summary":          build_rcm_summary(silver_charges, silver_patientvisits),
        # Enhanced Gold join — 3 conditions + discharged patients only
        "rcm_summary_v2":       build_rcm_summary_v2(silver_charges, silver_patientvisits),
    }
