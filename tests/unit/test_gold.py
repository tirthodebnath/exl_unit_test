"""
test_gold.py
============
Unit tests for the Gold layer — LEFT JOIN charges to patientvisits.

PURPOSE:
    These tests run on DUMMY DATA before the real pipeline executes.
    They verify that the Gold join produces correct results — the right
    rows, the right columns, the right amounts, and the right join type.

ABOUT THE JOIN:
    Gold LEFT JOINs silver_charges to silver_patientvisits on rcm_client_id.
    This is a one-to-many join — one visit can have many charge lines.
    Example: EXL001 has 4 charges and 2 visits → 4×2 = 8 Gold rows.

    Because of this fan-out, all reconciliation tests work on DISTINCT
    charge_ids in Gold, not raw row counts.

DATA SOURCE:
    The silver_and_gold fixture builds Silver charges (5 rows),
    Silver patientvisits (3 rows), and Gold output entirely in memory.
    No files or Delta tables are used.
"""

import pytest
from decimal import Decimal
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DecimalType

from src.silver.transform_charges import build_silver_charges
from src.silver.transform_patientvisits import build_silver_patientvisits
from src.gold.aggregate import build_rcm_summary


# ---------------------------------------------------------------------------
# Shared fixture: builds Silver and Gold from the Bronze fixtures in conftest
# ---------------------------------------------------------------------------
@pytest.fixture
def silver_and_gold(bronze_charges_df, bronze_patientvisits_df):
    """
    Build Silver charges, Silver patientvisits, and Gold output in one step.

    WHY A FIXTURE:
        All Gold tests need the same Silver + Gold DataFrames. Putting this
        in a fixture avoids duplicating 4 lines of setup in every test method.

    Returns:
        tuple: (silver_charges DataFrame, gold DataFrame)

    NOTE:
        silver_charges has 5 rows (CHG001–CHG005)
        silver_patientvisits has 3 rows (EXL001/PA001, EXL001/PA002, EXL002/PA003)
        gold has 9 rows (EXL001: 4 charges × 2 visits = 8, EXL002: 1×1 = 1)
    """
    # Build Silver from dummy Bronze fixtures defined in conftest.py
    silver_charges       = build_silver_charges(bronze_charges_df)
    silver_patientvisits = build_silver_patientvisits(bronze_patientvisits_df)

    # Build Gold by LEFT JOINing charges to patientvisits on rcm_client_id
    gold = build_rcm_summary(silver_charges, silver_patientvisits)

    return silver_charges, gold


# ---------------------------------------------------------------------------
# TestGoldReconciliation: Data integrity between Silver and Gold
# ---------------------------------------------------------------------------
class TestGoldReconciliation:
    """
    Reconciliation tests confirm that no data is created, dropped, or
    corrupted when Silver is promoted to Gold through the LEFT JOIN.

    All 4 tests use DISTINCT charge_ids in Gold to account for the
    one-to-many fan-out (same charge appears once per matching visit).
    """

    def test_charge_count_matches_silver(self, silver_and_gold):
        """
        WHAT: Distinct charge_ids in Gold must equal row count in Silver.

        WHY: If a charge is silently dropped at the join step, reports will
             undercount revenue. If a phantom charge is invented, revenue
             is overstated. Counting distinct charge_ids in Gold and comparing
             to Silver row count catches both defects.

        VALIDATES:
            - gold.distinct(charge_id).count() == silver.count()
        """
        silver, gold = silver_and_gold

        silver_count      = silver.count()
        gold_distinct_cnt = gold.select("charge_id").distinct().count()

        assert gold_distinct_cnt == silver_count, (
            f"Silver has {silver_count} charges but Gold has "
            f"{gold_distinct_cnt} distinct charge_ids"
        )

    def test_total_charge_amount_matches_silver(self, silver_and_gold):
        """
        WHAT: Sum of charge amounts in Gold (deduped on charge_id) must
              equal the sum in Silver.

        WHY: The one-to-many join fans out rows — each charge appears once
             per matching visit. If we summed amounts from all Gold rows
             without deduping first, we would overcount. This test:
               1. Dedupes Gold on charge_id to remove fan-out copies
               2. Sums the deduped amounts
               3. Compares to Silver total
             Any difference means money was lost or invented at the join step.

        VALIDATES:
            - gold.dropDuplicates(["charge_id"]).sum(amount) == silver.sum(amount)
        """
        silver, gold = silver_and_gold

        # Sum amounts directly in Silver (each charge appears once)
        silver_total = silver.agg(F.sum("amount").alias("t")).first()["t"]

        # Dedupe Gold on charge_id first — removes fan-out duplicates
        # Then sum the unique charge amounts
        gold_total = (
            gold.dropDuplicates(["charge_id"])
                .agg(F.sum("amount").alias("t"))
                .first()["t"]
        )

        assert silver_total == gold_total, (
            f"Amount mismatch after join — Silver total: {silver_total}, "
            f"Gold total (charge_id deduped): {gold_total}"
        )

    def test_no_new_charge_ids_introduced_in_gold(self, silver_and_gold):
        """
        WHAT: Every charge_id in Gold must also exist in Silver.
              The join must never invent charge_ids.

        WHY: If Gold contains a charge_id that Silver does not have, it means
             the join accidentally created data from nowhere. This is a severe
             defect that would inflate revenue reports with phantom charges.

        VALIDATES:
            - set(gold charge_ids) - set(silver charge_ids) == empty set
        """
        silver, gold = silver_and_gold

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        # Any ID in Gold that is NOT in Silver is a phantom — must be zero
        new_ids = gold_ids - silver_ids
        assert len(new_ids) == 0, (
            f"Gold contains charge_ids not found in Silver: {new_ids}"
        )

    def test_no_silver_charge_ids_dropped_in_gold(self, silver_and_gold):
        """
        WHAT: Every charge_id in Silver must appear in Gold.
              No charge should be silently dropped by the join.

        WHY: This is the most critical Gold test. If someone accidentally
             changes the LEFT JOIN to an INNER JOIN, charges that have no
             matching visit record would silently disappear. In RCM, those
             are exactly the unreconciled charges that need investigation —
             they must never be hidden.

        VALIDATES:
            - set(silver charge_ids) - set(gold charge_ids) == empty set
        """
        silver, gold = silver_and_gold

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        # Any Silver ID missing from Gold means LEFT JOIN became INNER JOIN
        dropped_ids = silver_ids - gold_ids
        assert len(dropped_ids) == 0, (
            f"Silver charge_ids are missing from Gold (possible INNER JOIN bug): "
            f"{dropped_ids}"
        )


# ---------------------------------------------------------------------------
# TestGoldJoinCorrectness: Verify the join type and structure are correct
# ---------------------------------------------------------------------------
class TestGoldJoinCorrectness:
    """
    Tests that confirm the join itself is working correctly —
    the right type (LEFT), the right key (rcm_client_id), and
    the right column structure (pv_ prefix on visit columns).
    """

    def test_left_join_keeps_charge_without_matching_visit(
        self, spark, make_charge, make_patientvisit,
        bronze_charges_schema, bronze_patientvisits_schema
    ):
        """
        WHAT: A charge for rcm_client_id EXL999 has no matching visit.
              After the LEFT JOIN, that charge must still be in Gold with
              null values for all visit columns.

        WHY: In RCM, unreconciled charges (no matching visit) represent
             billing gaps that need follow-up. They must NEVER disappear
             silently from Gold. If they do, the billing team has no way
             to know they exist.

        VALIDATES:
            - Orphan charge (no matching visit) survives in Gold
            - Visit columns are null for the orphan charge
        """
        # Charge for a client that has NO visit in the patientvisits table
        orphan_charge = make_charge(
            rcm_client_id="EXL999",    # no matching visit for EXL999
            charge_id="CHG_ORPHAN"
        )
        # Visit only exists for EXL001 — EXL999 has no visit
        visit = make_patientvisit(rcm_client_id="EXL001")

        charges_df = spark.createDataFrame([orphan_charge],
                                           schema=bronze_charges_schema)
        visits_df  = spark.createDataFrame([visit],
                                           schema=bronze_patientvisits_schema)

        sc  = build_silver_charges(charges_df)
        spv = build_silver_patientvisits(visits_df)
        out = build_rcm_summary(sc, spv)

        # Orphan charge must be present in Gold
        orphan_row = out.filter("charge_id = 'CHG_ORPHAN'").first()
        assert orphan_row is not None, (
            "Charge with no matching visit was dropped by the join — "
            "must use LEFT JOIN not INNER JOIN"
        )

        # Visit columns must be null — no visit exists for EXL999
        assert orphan_row["pv_patient_account_number"] is None, (
            "Visit columns must be null for charges with no matching visit"
        )

    def test_visit_columns_are_prefixed_with_pv(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: All columns that originated from the patientvisits table
              must have a pv_ prefix in the Gold output.

        WHY: Both tables share column names (patient_account_number, org_code,
             etc). Without the pv_ prefix, the join would produce ambiguous
             column names making it impossible to know which table a column
             came from. The pv_ prefix also makes the Gold output self-documenting.

        VALIDATES:
            - pv_patient_account_number exists in Gold
            - pv_computed_los_days exists in Gold
            - pv_has_insurance_balance exists in Gold
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # These are the key pv_-prefixed columns we expect
        expected_pv_cols = [
            "pv_patient_account_number",
            "pv_computed_los_days",
            "pv_has_insurance_balance",
        ]
        for col in expected_pv_cols:
            assert col in out.columns, (
                f"Expected pv_-prefixed column '{col}' not found in Gold. "
                f"Visit columns must be prefixed to avoid ambiguity."
            )

    def test_rcm_client_id_appears_exactly_once(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: rcm_client_id must appear exactly once in the Gold output —
              not duplicated as rcm_client_id + pv_rcm_client_id.

        WHY: rcm_client_id is the join key. When joining, Spark returns
             the join key from one side only (not both) to avoid redundancy.
             Our aggregate.py explicitly handles this by prefixing visit cols
             and using rcm_client_id as the ON key. This test confirms the
             column appears exactly once.

        VALIDATES:
            - rcm_client_id appears exactly once in Gold column list
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # Count occurrences of rcm_client_id in column list
        occurrences = out.columns.count("rcm_client_id")
        assert occurrences == 1, (
            f"rcm_client_id appears {occurrences} times in Gold columns. "
            f"Expected exactly 1."
        )

    def test_one_to_many_row_count_is_correct(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: Verify the one-to-many join produces the mathematically
              correct number of rows.

        WHY: Understanding the fan-out is critical. With 4 EXL001 charges
             and 2 EXL001 visits, the join produces 4×2=8 rows for EXL001.
             With 1 EXL002 charge and 1 EXL002 visit, the join produces 1 row.
             Total = 9. If the count is wrong, the join logic is broken.

        VALIDATES:
            - Total Gold row count = (EXL001 charges × EXL001 visits) + EXL002
            - Confirms one-to-many fan-out is working as designed
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # EXL001: 4 charges × 2 visits = 8 rows
        # EXL002: 1 charge × 1 visit = 1 row
        # Total = 9
        expected_rows = 9
        actual_rows   = out.count()

        assert actual_rows == expected_rows, (
            f"One-to-many join produced wrong row count. "
            f"Expected {expected_rows}, got {actual_rows}"
        )


# ---------------------------------------------------------------------------
# TestGoldDataAccuracy: Verify that Silver data survives the join unchanged
# ---------------------------------------------------------------------------
class TestGoldDataAccuracy:
    """
    Tests that confirm charge data is not corrupted or altered by the
    Gold LEFT JOIN. The join must only ADD visit columns — never change
    charge column values.
    """

    def test_amount_band_survives_join(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: The amount_band column computed in Silver must be unchanged
              in the Gold output.

        WHY: amount_band is a business-critical derived column. If it is
             altered by the join, amount-based filtering and reporting
             (LOW/MEDIUM/HIGH/JUMBO breakdowns) would be wrong in Gold.

        VALIDATES:
            - CHG001 (£250 → MEDIUM) has amount_band = MEDIUM in Gold
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # CHG001 has amount 250.00 which falls in MEDIUM band (100 < x <= 1000)
        chg001_rows = out.filter("charge_id = 'CHG001'").collect()

        # There may be multiple Gold rows for CHG001 due to fan-out
        # All copies must have the same amount_band
        for row in chg001_rows:
            assert row["amount_band"] == "MEDIUM", (
                f"CHG001 amount_band was altered by the join. "
                f"Expected MEDIUM, got {row['amount_band']}"
            )

    def test_charge_amount_not_altered_by_join(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: The actual amount value of a charge must be identical in
              Gold to what it was in Silver.

        WHY: If the join somehow modified amounts (e.g. through accidental
             aggregation or column aliasing), revenue reporting would be
             fundamentally broken. This test provides an explicit guarantee
             that amounts pass through unchanged.

        VALIDATES:
            - CHG001 amount in Gold equals CHG001 amount in Silver (£250.00)
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # CHG001 amount must be exactly 250.00 in Gold
        gold_amounts = {
            row["amount"]
            for row in out.filter("charge_id = 'CHG001'").select("amount").collect()
        }

        # All copies of CHG001 in Gold (due to fan-out) must have same amount
        assert len(gold_amounts) == 1, (
            f"CHG001 has different amounts in different Gold rows: {gold_amounts}"
        )
        assert Decimal("250.00") in gold_amounts, (
            f"CHG001 amount was changed by the join. "
            f"Expected 250.00, got {gold_amounts}"
        )

    def test_service_date_type_preserved_in_gold(
        self, bronze_charges_df, bronze_patientvisits_df
    ):
        """
        WHAT: service_date must remain DateType in Gold after the join.

        WHY: Spark joins occasionally cause type coercion when column types
             conflict between the two sides. service_date must stay as DateType
             so Gold consumers can do date arithmetic (e.g. days since service)
             without additional casting.

        VALIDATES:
            - service_date in Gold output schema is DateType
        """
        sc  = build_silver_charges(bronze_charges_df)
        spv = build_silver_patientvisits(bronze_patientvisits_df)
        out = build_rcm_summary(sc, spv)

        # service_date must be DateType — not String or Timestamp
        assert isinstance(out.schema["service_date"].dataType, DateType), (
            f"service_date type was changed by the join. "
            f"Expected DateType, got {out.schema['service_date'].dataType}"
        )
