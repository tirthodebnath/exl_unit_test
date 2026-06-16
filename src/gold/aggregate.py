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


def build_ogom_charges(
    silver_charges: DataFrame,
    silver_patientvisits: DataFrame,
) -> DataFrame:
    """
    Build the OGOM Charges Gold table.

    WHAT IT DOES:
        Joins Silver charges to Silver patientvisits using the same
        conditions as build_rcm_summary_v2, then selects and computes
        only the columns relevant to charges and patientvisits.

        Maps to the SQL that creates:
            prod_lca_unrestricted.bassett_epic_acute_gold.ogomcharges

    WHAT IS KEPT (charges + patientvisits columns only):
        From charges:
            rcm_client_id, rcm_npi, rcm_patient_account_number,
            rcm_unique_account_number, org_code, patient_account_number,
            host_system, charge_code, charge_code_description, amount,
            charge_posting_date, health_plan_number, financial_class_at_txn,
            file_date, service_date, charge_id, revenue_code, hcpcs_cpt_code,
            modifier1, modifier2, corrected_charge_flag, national_drug_code,
            department, patient_type_code

        From patientvisits (prefixed pv_):
            patient_admission_date, patient_discharge_date,
            attending_provider_code, service_code

    WHAT IS SKIPPED (requires lookup tables we do not have):
        charge_aging_category   → needs acute_mncragingstratdictionary (asd)
        ogom_patient_type       → needs acute_patienttypecodedictionary (ptd)
        payer_name              → needs acute_healthplandictionary (hpd)
        client_financial_class  → needs hpd
        ogom_payer_rollup       → needs hpd

    COMPUTED COLUMNS (derived from charges + patientvisits only):
        ogom_transaction_type
            Always "Charge" — literal value marking this as a charge record.

        charge_age
            DATEDIFF(charge_posting_date, patient_discharge_date)
            NULL when patient_discharge_date is NULL (not yet discharged).
            Measures how many days after discharge the charge was posted.

        late_charge_flag
            NULL  when patient_discharge_date is NULL
            1     when outpatient/emergency (O/E) and charge_age > 5 days
            1     when inpatient (I) and charge_age > 3 days
            0     otherwise
            Uses c.patient_type_code directly (ptd lookup table skipped).

        charge_lag_days
            DATEDIFF(charge_posting_date, service_date)
            NULL when either date is NULL.
            Measures how many days between service and charge being posted.

        charge_capture_days
            DATEDIFF(charge_posting_date, patient_admission_date)
            NULL when either date is NULL.
            Measures how many days between admission and charge capture.

    JOIN CONDITIONS (same as build_rcm_summary_v2):
        charges.rcm_client_id          = patientvisits.rcm_client_id
        charges.rcm_npi                = patientvisits.rcm_npi
        charges.patient_account_number = patientvisits.patient_account_number
        patientvisits.patient_discharge_date IS NOT NULL

    COLUMN NAME MAPPING (original SQL alias → our column name):
        ica_client_id                  → rcm_client_id
        Ica_NPI                        → rcm_npi
        lca_PatientAccountNumber       → rcm_patient_account_number
        lca_UniqueAccountNumber        → rcm_unique_account_number
        ClientFacilityCode             → org_code
        PatientAccountNumber           → patient_account_number
        AdmitDate                      → pv_patient_admission_date
        DischargeDate                  → pv_patient_discharge_date
        HospitalServiceCode            → pv_service_code
        AttendingProviderID            → pv_attending_provider_code
        ChargeAmount                   → amount
        PostingDate                    → charge_posting_date
        DepartmentNumber               → department

    Args:
        silver_charges       (DataFrame): Cleaned charges from Silver layer.
        silver_patientvisits (DataFrame): Cleaned visits from Silver layer.

    Returns:
        DataFrame: OGOM Charges Gold table with selected and computed columns.
                   All charges preserved — LEFT JOIN guarantee.
                   Writes to: workspace.tirtho_db.gold_ogom_charges
    """

    # ── Step 1: Filter visits — only discharged patients are join-eligible ──
    # Maps to: AND pv.PatientDischargeDate IS NOT NULL in the ON clause.
    # Visits without a discharge date are excluded before joining.
    # Charges whose visit has no discharge date get null pv_ columns.
    discharged_visits = silver_patientvisits.filter(
        F.col("patient_discharge_date").isNotNull()
    )

    # ── Step 2: Select only the visit columns we need ───────────────────────
    # We do not need all 118 patientvisits columns in Gold.
    # Only these 4 are used in the final SELECT or computed columns.
    pv_needed = discharged_visits.select(
        "rcm_client_id",              # join key — kept unprefixed
        "rcm_npi",                    # join key — kept unprefixed for join
        "patient_account_number",     # join key — kept unprefixed for join
        "patient_admission_date",     # used in charge_capture_days
        "patient_discharge_date",     # used in charge_age, late_charge_flag
        "attending_provider_code",    # AttendingProviderID in SQL
        "service_code",               # HospitalServiceCode in SQL
    ).select(
        # Keep rcm_client_id unprefixed — used as join key
        F.col("rcm_client_id"),
        # Prefix everything else to avoid column name clashes with charges
        F.col("rcm_npi").alias("pv_rcm_npi"),
        F.col("patient_account_number").alias("pv_patient_account_number"),
        F.col("patient_admission_date").alias("pv_patient_admission_date"),
        F.col("patient_discharge_date").alias("pv_patient_discharge_date"),
        F.col("attending_provider_code").alias("pv_attending_provider_code"),
        F.col("service_code").alias("pv_service_code"),
    )

    # ── Step 3: LEFT JOIN charges to discharged visits ──────────────────────
    # All 3 conditions from the original SQL must match for a visit to link.
    # LEFT JOIN: charges with no matching visit stay in output with null pv_ cols.
    joined = silver_charges.join(
        pv_needed,
        on=(
            # Condition 1: same client (c.lca_client_id = pv.lca_client_id)
            (silver_charges["rcm_client_id"] == pv_needed["rcm_client_id"])
            # Condition 2: same NPI (c.lca_NPI = pv.lca_NPI)
            & (silver_charges["rcm_npi"] == pv_needed["pv_rcm_npi"])
            # Condition 3: same patient account
            & (silver_charges["patient_account_number"] == pv_needed["pv_patient_account_number"])
        ),
        how="left",
    ).drop(pv_needed["rcm_client_id"])  # drop duplicate join key from visit side

    # ── Step 4: Select and compute final output columns ─────────────────────
    return joined.select(

        # ── Identity columns from charges ───────────────────────────────────
        # Maps to: c.lca_client_id AS ica_client_id
        F.col("rcm_client_id"),
        # Maps to: c.lca_NPI AS Ica_NPI
        F.col("rcm_npi"),
        # Maps to: c.lca_PatientAccountNumber AS lca_PatientAccountNumber
        F.col("rcm_patient_account_number"),
        # Maps to: c.lca_UniqueAccountNumber AS lca_UniqueAccountNumber
        F.col("rcm_unique_account_number"),
        # Maps to: c.OrgCode AS ClientFacilityCode
        F.col("org_code"),
        # Maps to: c.PatientAccountNumber AS PatientAccountNumber
        F.col("patient_account_number"),
        # Maps to: c.HostSystem AS HostSystem
        F.col("host_system"),

        # ── Visit date columns ───────────────────────────────────────────────
        # Maps to: pv.PatientAdmissionDate AS AdmitDate
        F.col("pv_patient_admission_date").alias("admit_date"),
        # Maps to: pv.PatientDischargeDate AS DischargeDate
        F.col("pv_patient_discharge_date").alias("discharge_date"),

        # ── Computed: ChargeAge ──────────────────────────────────────────────
        # Maps to:
        #   CASE WHEN pv.PatientDischargeDate IS NULL THEN NULL
        #        ELSE DATEDIFF(c.ChargePostingDate, pv.PatientDischargeDate)
        #   END AS ChargeAge
        # WHY: measures how many days after discharge the charge was posted.
        # NULL when not yet discharged (discharge date null from LEFT JOIN).
        F.when(
            F.col("pv_patient_discharge_date").isNull(),
            F.lit(None)
        ).otherwise(
            F.datediff(
                F.col("charge_posting_date"),
                F.col("pv_patient_discharge_date")
            )
        ).alias("charge_age"),

        # ── Literal: OGOMTransactionType ────────────────────────────────────
        # Maps to: "Charge" AS OGOMTransactionType
        # WHY: identifies this Gold table as containing charge transactions.
        # All rows always have this value — no conditions.
        F.lit("Charge").alias("ogom_transaction_type"),

        # ── Visit service columns ────────────────────────────────────────────
        # Maps to: pv.PatientTypeCode AS PatientTypeCode
        # Using charge's patient_type_code (ptd lookup skipped)
        F.col("patient_type_code"),
        # Maps to: pv.ServiceCode AS HospitalServiceCode
        F.col("pv_service_code").alias("hospital_service_code"),

        # ── Charge detail columns ────────────────────────────────────────────
        F.col("charge_code"),
        F.col("charge_code_description"),
        # Maps to: c.Amount AS ChargeAmount
        F.col("amount").alias("charge_amount"),
        # Maps to: c.ChargePostingDate AS PostingDate
        F.col("charge_posting_date").alias("posting_date"),
        F.col("health_plan_number"),
        F.col("financial_class_at_txn"),
        # Maps to: pv.AttendingProviderCode AS AttendingProviderID
        F.col("pv_attending_provider_code").alias("attending_provider_id"),
        F.col("file_date"),

        # ── Computed: LateChargeFlag ─────────────────────────────────────────
        # Maps to:
        #   CASE WHEN pv.PatientDischargeDate IS NULL THEN NULL
        #        WHEN OGOMPatientType IN ('O','E') AND ChargeAge > 5 THEN 1
        #        WHEN OGOMPatientType = 'I' AND ChargeAge > 3 THEN 1
        #        ELSE 0
        #   END AS LateChargeFlag
        # WHY: identifies charges submitted after acceptable timelines.
        # Using c.patient_type_code directly (ptd lookup table skipped).
        F.when(
            F.col("pv_patient_discharge_date").isNull(), F.lit(None)
        ).when(
            # Outpatient/Emergency: late if posted > 5 days after discharge
            F.upper(F.col("patient_type_code")).isin("O", "E") &
            (F.datediff(F.col("charge_posting_date"),
                        F.col("pv_patient_discharge_date")) > 5),
            F.lit(1)
        ).when(
            # Inpatient: late if posted > 3 days after discharge
            F.upper(F.col("patient_type_code")) == F.lit("I") &
            (F.datediff(F.col("charge_posting_date"),
                        F.col("pv_patient_discharge_date")) > 3),
            F.lit(1)
        ).otherwise(F.lit(0)).alias("late_charge_flag"),

        # Maps to: c.ServiceDate AS ServiceDate
        F.col("service_date"),
        # Maps to: c.ChargeID AS ChargeID
        F.col("charge_id"),

        # ── Computed: ChargeLagDays ──────────────────────────────────────────
        # Maps to:
        #   CASE WHEN c.ChargePostingDate IS NOT NULL AND c.ServiceDate IS NOT NULL
        #        THEN DATEDIFF(c.ChargePostingDate, c.ServiceDate)
        #   END AS ChargeLagDays
        # WHY: measures delay between when service was rendered and posted.
        F.when(
            F.col("charge_posting_date").isNotNull() &
            F.col("service_date").isNotNull(),
            F.datediff(F.col("charge_posting_date"), F.col("service_date"))
        ).otherwise(F.lit(None)).alias("charge_lag_days"),

        # ── Computed: ChargeCaptureDays ──────────────────────────────────────
        # Maps to:
        #   CASE WHEN c.ChargePostingDate IS NOT NULL
        #        AND pv.PatientAdmissionDate IS NOT NULL
        #        THEN DATEDIFF(c.ChargePostingDate, pv.PatientAdmissionDate)
        #        ELSE NULL
        #   END AS ChargeCaptureDays
        # WHY: measures how many days after admission the charge was captured.
        F.when(
            F.col("charge_posting_date").isNotNull() &
            F.col("pv_patient_admission_date").isNotNull(),
            F.datediff(F.col("charge_posting_date"),
                       F.col("pv_patient_admission_date"))
        ).otherwise(F.lit(None)).alias("charge_capture_days"),

        # ── Remaining charge columns ─────────────────────────────────────────
        F.col("revenue_code"),
        F.col("hcpcs_cpt_code"),
        F.col("modifier1"),
        F.col("modifier2"),
        F.col("corrected_charge_flag"),
        F.col("national_drug_code"),
        # Maps to: c.Department AS DepartmentNumber
        F.col("department").alias("department_number"),
    )
