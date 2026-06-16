"""
tests/integration/gold/test_ogom_charges.py
============================================
PURPOSE:
    Data quality tests for workspace.tirtho_db.gold_ogom_charges.
    This is the ONLY Gold integration test file.
    Run by: notebooks/run_tests_gold.py

TESTS (6 total):
    1. test_no_new_charge_ids_in_gold
       From: test_charges.py test 3
       Mapped to gold_ogom_charges instead of gold_rcm_summary_v2

    2. test_no_silver_charge_ids_dropped
       From: test_charges.py test 4
       Mapped to gold_ogom_charges instead of gold_rcm_summary_v2

    3. test_ogom_transaction_type_always_charge
       From: test_ogom_charges.py test 5
       Every row must have ogom_transaction_type = 'Charge'

    4. test_charge_age_null_when_no_discharge
       From: test_ogom_charges.py test 6
       charge_age must be null when discharge_date is null

    5. test_discharge_date_after_admit_date  [NEW]
       Where both dates exist, discharge must be after admission.
       A patient cannot be discharged before being admitted.

    6. test_no_duplicate_join_combinations   [NEW]
       Maps to:
           SELECT rcm_client_id, rcm_npi, patient_account_number,
                  COUNT(*) AS cnt
           FROM gold_ogom_charges
           WHERE admit_date IS NOT NULL   -- actual_join_flag = 1
           GROUP BY rcm_client_id, rcm_npi, patient_account_number
           HAVING COUNT(*) > 1
       Must return zero rows — the 3-condition join must be 1-to-1.
       If any group has count > 1, the same (client + NPI + patient)
       combination matched multiple visit rows (fan-out).

TABLE USED:
    workspace.tirtho_db.gold_ogom_charges (primary)
    workspace.tirtho_db.silver_charges    (for tests 1, 2 only)
"""

import pytest
from pyspark.sql import functions as F


def _get_ogom_charges(spark):
    """
    Load the real gold_ogom_charges Delta table.

    WHY A HELPER:
        All 6 tests need the same table. One place to load it
        means if the table name changes, we change it once only.
        Also skips cleanly with a clear message if table is missing.

    Args:
        spark (SparkSession): Active Databricks session.

    Returns:
        DataFrame: All rows from workspace.tirtho_db.gold_ogom_charges.
    """
    try:
        return spark.table("workspace.tirtho_db.gold_ogom_charges")
    except Exception as e:
        pytest.skip(
            f"gold_ogom_charges not found. "
            f"Run run_pipeline.py first. Error: {e}"
        )


# ---------------------------------------------------------------------------
# TEST 1: No new charge_ids in Gold
# Moved from test_charges.py test 3 — mapped to gold_ogom_charges
# ---------------------------------------------------------------------------
def test_no_new_charge_ids_in_gold(spark):
    """
    WHAT: Every charge_id in gold_ogom_charges must also exist in
          silver_charges. The Gold join must never invent charge_ids.

    WHY: A phantom charge_id in Gold means revenue data was created
         from nowhere. Gold reports would show more charges than
         actually exist in Silver — a fundamental data integrity failure.

    ORIGINAL: test_charges.py test 3 — same logic, mapped from
              gold_rcm_summary_v2 to gold_ogom_charges.

    TABLE USED:
        gold_ogom_charges  — Gold table being tested
        silver_charges     — reference set of valid charge_ids

    REAL DATA CHECK:
        set(gold charge_ids) - set(silver charge_ids) == empty set
    """
    gold = _get_ogom_charges(spark)

    try:
        silver = spark.table("workspace.tirtho_db.silver_charges")
    except Exception as e:
        pytest.skip(f"silver_charges not available for comparison. Error: {e}")

    # Collect charge_ids from both tables
    silver_ids = {
        r["charge_id"]
        for r in silver.select("charge_id").collect()
    }
    gold_ids = {
        r["charge_id"]
        for r in gold.select("charge_id").distinct().collect()
    }

    # Any ID in Gold that is NOT in Silver is a phantom — must be zero
    phantom = gold_ids - silver_ids

    assert len(phantom) == 0, (
        f"gold_ogom_charges contains {len(phantom)} charge_id(s) "
        f"not found in silver_charges: {phantom}. "
        f"The Gold join must never invent charge_ids."
    )


# ---------------------------------------------------------------------------
# TEST 2: No Silver charge_ids dropped from Gold
# Moved from test_charges.py test 4 — mapped to gold_ogom_charges
# ---------------------------------------------------------------------------
def test_no_silver_charge_ids_dropped(spark):
    """
    WHAT: Every charge_id in silver_charges must appear in
          gold_ogom_charges. No charge should be silently dropped.

    WHY: This is the most critical Gold test. The LEFT JOIN guarantees
         all charges appear regardless of whether a visit matched.
         If any Silver charge_id is missing from Gold, the LEFT JOIN
         accidentally became an INNER JOIN somewhere — unreconciled
         charges disappear from all billing reports.

    ORIGINAL: test_charges.py test 4 — same logic, mapped from
              gold_rcm_summary_v2 to gold_ogom_charges.

    TABLE USED:
        gold_ogom_charges — Gold table being tested
        silver_charges    — source of all charge_ids that must appear

    REAL DATA CHECK:
        set(silver charge_ids) - set(gold charge_ids) == empty set
    """
    gold = _get_ogom_charges(spark)

    try:
        silver = spark.table("workspace.tirtho_db.silver_charges")
    except Exception as e:
        pytest.skip(f"silver_charges not available for comparison. Error: {e}")

    silver_ids = {
        r["charge_id"]
        for r in silver.select("charge_id").collect()
    }
    gold_ids = {
        r["charge_id"]
        for r in gold.select("charge_id").distinct().collect()
    }

    # Any Silver ID missing from Gold = charge was dropped = critical defect
    dropped = silver_ids - gold_ids

    assert len(dropped) == 0, (
        f"{len(dropped)} silver_charges charge_id(s) are missing "
        f"from gold_ogom_charges: {dropped}. "
        f"LEFT JOIN must keep ALL charges even without a matching visit."
    )


# ---------------------------------------------------------------------------
# TEST 3: ogom_transaction_type is always 'Charge'
# From test_ogom_charges.py test 5 — unchanged
# ---------------------------------------------------------------------------
def test_ogom_transaction_type_always_charge(spark):
    """
    WHAT: ogom_transaction_type must be 'Charge' on every single row
          in gold_ogom_charges.

    WHY: Maps to the literal "Charge" AS OGOMTransactionType in the
         original SQL. This column identifies all rows in this table
         as charge transactions. If any row has a different value,
         the literal computation failed or wrong data was loaded.

    FROM: test_ogom_charges.py test 5 — same assertion, same table.

    TABLE USED:
        gold_ogom_charges

    REAL DATA CHECK:
        All distinct ogom_transaction_type values == {"Charge"}
    """
    df = _get_ogom_charges(spark)

    distinct_types = {
        r["ogom_transaction_type"]
        for r in df.select("ogom_transaction_type").distinct().collect()
    }

    assert distinct_types == {"Charge"}, (
        f"ogom_transaction_type has unexpected values: {distinct_types}. "
        f"Every row in gold_ogom_charges must have exactly 'Charge'."
    )


# ---------------------------------------------------------------------------
# TEST 4: charge_age is null when discharge_date is null
# From test_ogom_charges.py test 6 — unchanged
# ---------------------------------------------------------------------------
def test_charge_age_null_when_no_discharge(spark):
    """
    WHAT: charge_age must be null whenever discharge_date is null.

    WHY: Maps to:
             CASE WHEN pv.PatientDischargeDate IS NULL THEN NULL
                  ELSE DATEDIFF(ChargePostingDate, PatientDischargeDate)
             END AS ChargeAge
         A charge cannot have an age relative to a discharge that has
         not happened yet. If charge_age is not null when discharge_date
         is null, the CASE logic was not applied correctly.

    FROM: test_ogom_charges.py test 6 — same assertion, same table.

    TABLE USED:
        gold_ogom_charges

    REAL DATA CHECK:
        Zero rows where discharge_date IS NULL AND charge_age IS NOT NULL.
    """
    df = _get_ogom_charges(spark)

    violated = df.filter(
        F.col("discharge_date").isNull() &
        F.col("charge_age").isNotNull()
    ).count()

    assert violated == 0, (
        f"{violated} rows in gold_ogom_charges have null discharge_date "
        f"but non-null charge_age. "
        f"charge_age must be null when no discharge date exists."
    )


# ---------------------------------------------------------------------------
# TEST 5: discharge_date must be after admit_date [NEW]
# ---------------------------------------------------------------------------
def test_discharge_date_after_admit_date(spark):
    """
    WHAT: Where both admit_date and discharge_date are present,
          discharge_date must always be after (greater than) admit_date.

    WHY: A patient cannot be discharged before being admitted.
         If discharge_date <= admit_date on any row, it indicates:
           - A data entry error in the source system
           - A date mapping error in the pipeline
           - The wrong visit was joined to this charge
         Any of these corrupt AR aging, LOS calculations, and
         the late_charge_flag computation which depends on discharge.

    NEW TEST — not in any previous version.

    TABLE USED:
        gold_ogom_charges

    REAL DATA CHECK:
        Zero rows where both dates exist AND discharge_date <= admit_date.
    """
    df = _get_ogom_charges(spark)

    # Only check rows where BOTH dates are present
    # Rows with null admit_date or null discharge_date are not applicable
    violated = df.filter(
        F.col("admit_date").isNotNull() &
        F.col("discharge_date").isNotNull() &
        (F.col("discharge_date") <= F.col("admit_date"))
    ).count()

    assert violated == 0, (
        f"{violated} rows in gold_ogom_charges have discharge_date "
        f"on or before admit_date. "
        f"A patient must be admitted before they can be discharged."
    )


# ---------------------------------------------------------------------------
# TEST 6: No duplicate join combinations [NEW]
# Maps to the SQL:
#   SELECT rcm_client_id, rcm_npi, patient_account_number, COUNT(*) AS cnt
#   FROM gold_ogom_charges
#   WHERE admit_date IS NOT NULL      -- actual_join_flag = 1 (visit was joined)
#   GROUP BY rcm_client_id, rcm_npi, patient_account_number
#   HAVING COUNT(*) > 1
# Must return zero rows.
# ---------------------------------------------------------------------------
def test_no_duplicate_join_combinations(spark):
    """
    WHAT: After filtering for rows where a visit was successfully joined
          (admit_date IS NOT NULL), group by the 3 join keys and check
          no combination appears more than once.

    WHY: The Gold V2 join uses 3 conditions:
             rcm_client_id = rcm_client_id
             rcm_npi = rcm_npi
             patient_account_number = patient_account_number
         This should produce a 1-to-1 match — one charge links to at
         most one visit. If any (client + NPI + patient_account) group
         has COUNT(*) > 1 in the joined rows, it means:
           - The same patient had multiple visits matching all 3 conditions
           - The join produced fan-out (same charge linked to multiple visits)
           - Revenue would be double-counted in charge-level reports

    ORIGINAL SQL MAPPED TO OUR COLUMNS:
        client_id           → rcm_client_id
        NPI                 → rcm_npi
        PatientAccountNumber → patient_account_number
        actual_join_flag = 1 → admit_date IS NOT NULL

    NEW TEST — not in any previous version.

    TABLE USED:
        gold_ogom_charges

    REAL DATA CHECK:
        The query below must return zero rows:
            SELECT rcm_client_id, rcm_npi, patient_account_number,
                   COUNT(*) AS cnt
            FROM gold_ogom_charges
            WHERE admit_date IS NOT NULL
            GROUP BY rcm_client_id, rcm_npi, patient_account_number
            HAVING COUNT(*) > 1
    """
    df = _get_ogom_charges(spark)

    # Step 1: Filter to only rows where a visit was joined
    # admit_date IS NOT NULL means the LEFT JOIN found a matching visit
    # This maps to: WHERE actual_join_flag = 1 in the original SQL
    joined_rows = df.filter(F.col("admit_date").isNotNull())

    # Step 2: Group by the 3 join key columns and count occurrences
    # Maps to: GROUP BY client_id, NPI, PatientAccountNumber
    duplicate_groups = (
        joined_rows
        .groupBy("rcm_client_id", "rcm_npi", "patient_account_number")
        .agg(F.count("*").alias("cnt"))
        # Step 3: Keep only groups with more than 1 row
        # Maps to: HAVING COUNT(*) > 1
        .filter(F.col("cnt") > 1)
    )

    duplicate_count = duplicate_groups.count()

    assert duplicate_count == 0, (
        f"{duplicate_count} (rcm_client_id, rcm_npi, patient_account_number) "
        f"combination(s) appear more than once in joined rows. "
        f"The 3-condition join should be 1-to-1. "
        f"Fan-out detected — same charge linked to multiple visit rows."
    )
