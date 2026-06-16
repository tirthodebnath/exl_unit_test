"""
src/gold/aggregate.py
=====================
PURPOSE:
    Gold layer transformations — joins Silver charges to Silver patientvisits.

    Two join functions exist:
        build_rcm_summary()    — original join on rcm_client_id only (one-to-many)
        build_rcm_summary_v2() — enhanced join with 3 conditions + discharge filter
                                  maps to bassett_epic_acute_silver join logic

COLUMN MAPPING (bassett → our schema):
    c.client_id             → rcm_client_id
    c.NPI                   → rcm_npi
    c.PatientAccountNumber  → patient_account_number
    pv.PatientDischargeDate → patient_discharge_date
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_rcm_summary(
    silver_charges: DataFrame,
    silver_patientvisits: DataFrame,
) -> DataFrame:
    """
    Original Gold join — LEFT JOIN charges to patientvisits on rcm_client_id only.

    One-to-many: one visit row can match multiple charge rows for the same client.
    Visit columns are prefixed with pv_ to avoid column name ambiguity.

    Args:
        silver_charges       (DataFrame): Cleaned charges from Silver layer.
        silver_patientvisits (DataFrame): Cleaned visits from Silver layer.

    Returns:
        DataFrame: Charges with matched visit columns (pv_-prefixed).
                   All charges are preserved — LEFT JOIN guarantee.
    """
    # Prefix all visit columns except the join key to avoid column name clashes
    # Both tables share column names like patient_account_number, rcm_npi etc.
    visit_cols = [c for c in silver_patientvisits.columns if c != "rcm_client_id"]

    pv = silver_patientvisits.select(
        F.col("rcm_client_id"),
        *[F.col(c).alias(f"pv_{c}") for c in visit_cols]
    )

    # LEFT JOIN: every charge appears in output regardless of whether a visit matches
    return silver_charges.join(pv, on="rcm_client_id", how="left")


def build_rcm_summary_v2(
    silver_charges: DataFrame,
    silver_patientvisits: DataFrame,
) -> DataFrame:
    """
    Enhanced Gold join with 3 match conditions + discharge date filter.

    Maps to this SQL from bassett_epic_acute_silver:
        FROM charges c
        LEFT JOIN patientvisits pv
            ON  c.client_id            = pv.client_id            -- rcm_client_id
            AND c.NPI                  = pv.NPI                  -- rcm_npi
            AND c.PatientAccountNumber = pv.PatientAccountNumber  -- patient_account_number
            AND pv.PatientDischargeDate IS NOT NULL

    WHY MORE CONDITIONS THAN V1:
        The original join on rcm_client_id alone is one-to-many and can
        produce multiple visit rows per charge (fan-out). This join pins
        the match to a specific patient account AND NPI, giving a more
        precise charge-to-visit relationship. The discharge date filter
        ensures only completed (discharged) visits are considered.

    DISCHARGE DATE FILTER:
        Applied BEFORE the join — only discharged visits are eligible to match.
        Charges whose patient visit has a null discharge date will have null
        pv_ columns in the output (they are NOT dropped — LEFT JOIN guarantee).

    Args:
        silver_charges       (DataFrame): Cleaned charges from Silver layer.
        silver_patientvisits (DataFrame): Cleaned visits from Silver layer.

    Returns:
        DataFrame: Charges with precisely matched visit columns (pv_-prefixed).
                   All charges preserved regardless of visit match.
    """
    # ── Step 1: Filter visits to discharged patients only ──────────────────
    # pv.PatientDischargeDate IS NOT NULL → patient has been discharged
    # Un-discharged visits are not eligible to match any charge
    discharged_visits = silver_patientvisits.filter(
        F.col("patient_discharge_date").isNotNull()
    )

    # ── Step 2: Prefix all visit columns except rcm_client_id ──────────────
    # rcm_client_id is kept unprefixed as it is used in the join ON clause
    # All other columns get pv_ prefix to avoid ambiguity with charge columns
    visit_cols = [
        c for c in discharged_visits.columns
        if c != "rcm_client_id"
    ]
    pv = discharged_visits.select(
        F.col("rcm_client_id"),
        *[F.col(c).alias(f"pv_{c}") for c in visit_cols]
    )

    # ── Step 3: LEFT JOIN on 3 conditions ──────────────────────────────────
    # All 3 must match for a visit row to be linked to a charge row.
    # If no visit matches all 3 conditions, the charge still appears
    # in the output with null pv_ columns — LEFT JOIN guarantee.
    joined = silver_charges.join(
        pv,
        on=(
            # Condition 1: same client (was c.client_id = pv.client_id)
            (silver_charges["rcm_client_id"] == pv["rcm_client_id"])
            # Condition 2: same NPI (was c.NPI = pv.NPI)
            & (silver_charges["rcm_npi"] == pv["pv_rcm_npi"])
            # Condition 3: same patient account (was c.PatientAccountNumber = pv.PatientAccountNumber)
            & (silver_charges["patient_account_number"] == pv["pv_patient_account_number"])
        ),
        how="left",
    )

    # Drop the duplicate rcm_client_id from the visit side
    # (both sides have it — keep only the charge side version)
    return joined.drop(pv["rcm_client_id"])
