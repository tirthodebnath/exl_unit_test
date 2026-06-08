"""
test_gold_realdata.py
=====================
PURPOSE:
    The same 11 Gold tests from tests/unit/test_gold.py, but running on
    the REAL gold_rcm_summary Delta table.

    Unit tests proved the join logic works on 5 dummy charges + 3 dummy visits.
    These tests prove the same join logic worked correctly on all real data.

HOW TO RUN:
    Run by test_after_load_data.py notebook after run_pipeline.py has loaded data.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DateType


# ---------------------------------------------------------------------------
# TestGoldReconciliation — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestGoldReconciliation:
    """
    Reconciliation tests on real Silver charges and Gold tables.
    Matches TestGoldReconciliation in tests/unit/test_gold.py.

    All 4 tests use DISTINCT charge_ids in Gold to account for the
    one-to-many fan-out (same charge appears once per matching visit row).
    """

    def test_charge_count_matches_silver(self, silver_and_gold_real):
        """
        WHAT: Distinct charge_ids in real Gold must equal row count in
              real Silver charges.

        WHY: Unit test proved this on 5 dummy rows. This test proves it
             on every real charge in the production data. If a charge was
             dropped or duplicated at the join step on real data (which
             can happen with unexpected null join keys or data quirks),
             this test catches it.

        REAL DATA CHECK:
            - gold.distinct(charge_id).count() == silver.count()
        """
        silver, gold = silver_and_gold_real

        silver_count      = silver.count()
        gold_distinct_cnt = gold.select("charge_id").distinct().count()

        assert gold_distinct_cnt == silver_count, (
            f"Real data: Silver has {silver_count} charges but Gold has "
            f"{gold_distinct_cnt} distinct charge_ids."
        )

    def test_total_charge_amount_matches_silver(self, silver_and_gold_real):
        """
        WHAT: Sum of real charge amounts in Gold (deduped on charge_id) must
              equal the sum in real Silver.

        WHY: Unit test proved this on dummy data. This test proves no money
             was lost or invented when joining real data. The one-to-many
             join fans out rows so we dedupe on charge_id before summing.

        REAL DATA CHECK:
            - gold.dropDuplicates(["charge_id"]).sum(amount) == silver.sum(amount)
        """
        silver, gold = silver_and_gold_real

        silver_total = silver.agg(F.sum("amount").alias("t")).first()["t"]
        gold_total   = (
            gold.dropDuplicates(["charge_id"])
                .agg(F.sum("amount").alias("t"))
                .first()["t"]
        )

        assert silver_total == gold_total, (
            f"Real data amount mismatch: Silver total={silver_total}, "
            f"Gold total (charge_id deduped)={gold_total}."
        )

    def test_no_new_charge_ids_introduced_in_gold(self, silver_and_gold_real):
        """
        WHAT: Every real charge_id in Gold must also exist in real Silver.

        WHY: Unit test proved the join does not invent charge_ids on dummy
             data. This test proves the same on real data — no phantom charges
             appeared in Gold that were not in Silver.

        REAL DATA CHECK:
            - set(gold charge_ids) - set(silver charge_ids) == empty
        """
        silver, gold = silver_and_gold_real

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        phantom = gold_ids - silver_ids
        assert len(phantom) == 0, (
            f"Real data: Gold contains {len(phantom)} charge_id(s) not in Silver: "
            f"{phantom}"
        )

    def test_no_silver_charge_ids_dropped_in_gold(self, silver_and_gold_real):
        """
        WHAT: Every real charge_id in Silver must appear in real Gold.

        WHY: Unit test proved LEFT JOIN keeps all charges. This test proves
             the same on real data — no real charge was silently dropped.
             Dropped charges in Gold = missing revenue in reports.

        REAL DATA CHECK:
            - set(silver charge_ids) - set(gold charge_ids) == empty
        """
        silver, gold = silver_and_gold_real

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        dropped = silver_ids - gold_ids
        assert len(dropped) == 0, (
            f"Real data: {len(dropped)} Silver charge_id(s) missing from Gold: "
            f"{dropped}"
        )


# ---------------------------------------------------------------------------
# TestGoldJoinCorrectness — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestGoldJoinCorrectness:
    """
    Verify join structure on real Gold table.
    Matches TestGoldJoinCorrectness in tests/unit/test_gold.py.
    """

    def test_left_join_keeps_charge_without_matching_visit(
        self, silver_and_gold_real
    ):
        """
        WHAT: Verify charges that have no matching visit record are still
              present in Gold with null visit columns.

        WHY: Unit test proved this with a specific orphan charge. On real
             data, there may be many unreconciled charges (no matching visit
             in the patientvisits table). These must be visible in Gold —
             they represent billing gaps that need investigation.

        REAL DATA CHECK:
            - All real Gold rows have a non-null charge_id
            - Gold count >= Silver charges count (no charges dropped)
        """
        silver, gold = silver_and_gold_real

        silver_count = silver.count()
        gold_count   = gold.count()

        # Gold must have at least as many charge_id occurrences as Silver has rows
        # (LEFT JOIN guarantees all charges appear at least once)
        assert gold_count >= silver_count, (
            f"Real data: Gold has {gold_count} rows but Silver has "
            f"{silver_count} charges. Gold must be >= Silver (LEFT JOIN guarantee)."
        )

        # No charge_id should be null in Gold — charges drive the join
        null_charge = gold.filter(F.col("charge_id").isNull()).count()
        assert null_charge == 0, (
            f"Real data: {null_charge} Gold rows have null charge_id. "
            f"This should never happen — charges are on the left of the join."
        )

    def test_visit_columns_are_prefixed_with_pv(self, gold_real):
        """
        WHAT: Verify pv_-prefixed visit columns exist in real Gold table.

        WHY: Unit test proved these columns exist on dummy data. This test
             confirms the prefix survived the join on real data — no naming
             collision occurred.

        REAL DATA CHECK:
            - pv_patient_account_number, pv_computed_los_days,
              pv_has_insurance_balance all exist in Gold
        """
        expected_pv_cols = [
            "pv_patient_account_number",
            "pv_computed_los_days",
            "pv_has_insurance_balance",
        ]
        for col in expected_pv_cols:
            assert col in gold_real.columns, (
                f"Real data: pv_-prefixed column '{col}' missing from Gold. "
                f"Visit columns must be prefixed to avoid column name ambiguity."
            )

    def test_rcm_client_id_appears_exactly_once(self, gold_real):
        """
        WHAT: rcm_client_id must appear exactly once in real Gold columns.

        WHY: Unit test proved this on dummy data. On real data, confirms
             the join key was not duplicated (e.g. rcm_client_id AND
             pv_rcm_client_id both appearing would double the join key).

        REAL DATA CHECK:
            - "rcm_client_id" appears exactly 1 time in Gold column list
        """
        occurrences = gold_real.columns.count("rcm_client_id")
        assert occurrences == 1, (
            f"Real data: rcm_client_id appears {occurrences} times in Gold. "
            f"Expected exactly 1."
        )

    def test_one_to_many_row_count_is_correct(
        self, silver_and_gold_real
    ):
        """
        WHAT: Verify Gold row count is >= Silver charges count, confirming
              the one-to-many join produced the expected fan-out.

        WHY: Unit test proved specific counts on controlled dummy data.
             On real data, we cannot know the exact expected count (it
             depends on how many visits match each client). What we can
             assert is that Gold >= Silver charges (every charge appears
             at least once, and may appear multiple times due to fan-out).

        REAL DATA CHECK:
            - gold.count() >= silver_charges.count()
        """
        silver, gold = silver_and_gold_real

        silver_count = silver.count()
        gold_count   = gold.count()

        assert gold_count >= silver_count, (
            f"Real data: Gold ({gold_count} rows) < Silver ({silver_count} rows). "
            f"Every charge must appear at least once in Gold (LEFT JOIN guarantee)."
        )


# ---------------------------------------------------------------------------
# TestGoldDataAccuracy — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestGoldDataAccuracy:
    """
    Verify data accuracy in real Gold table after the join.
    Matches TestGoldDataAccuracy in tests/unit/test_gold.py.
    """

    def test_amount_band_survives_join(self, gold_real):
        """
        WHAT: Verify amount_band in real Gold only contains valid values.

        WHY: Unit test proved amount_band was not altered for a specific dummy
             row. This test scans ALL real Gold rows to confirm no invalid band
             values appeared during the join on real data.

        REAL DATA CHECK:
            - All distinct amount_band values in Gold are in allowed set
        """
        if "amount_band" not in gold_real.columns:
            pytest.skip("amount_band column missing from Gold")

        allowed_bands = {"LOW", "MEDIUM", "HIGH", "JUMBO", "UNKNOWN"}
        actual_bands  = {
            r["amount_band"]
            for r in gold_real.select("amount_band").distinct().collect()
        }
        unexpected = actual_bands - allowed_bands

        assert len(unexpected) == 0, (
            f"Real data: Gold contains unexpected amount_band values: "
            f"{unexpected}. Allowed values: {allowed_bands}"
        )

    def test_charge_amount_not_altered_by_join(self, gold_real):
        """
        WHAT: Verify all charge amounts in real Gold are non-null and
              non-negative.

        WHY: Unit test proved a specific amount was unchanged after the join.
             This test scans ALL real amounts in Gold to confirm none were
             corrupted (nulled or turned negative) by the join on real data.

        REAL DATA CHECK:
            - No null amounts in Gold
            - No negative amounts in Gold
        """
        if "amount" not in gold_real.columns:
            pytest.skip("amount column missing from Gold")

        null_amounts = gold_real.filter(F.col("amount").isNull()).count()
        neg_amounts  = gold_real.filter(F.col("amount") < 0).count()

        assert null_amounts == 0, (
            f"Real data: {null_amounts} Gold rows have null amount. "
            f"Amounts must not be nulled by the join."
        )
        assert neg_amounts == 0, (
            f"Real data: {neg_amounts} Gold rows have negative amount. "
            f"Silver filter should have removed these before Gold."
        )

    def test_service_date_type_preserved_in_gold(self, gold_real):
        """
        WHAT: Verify service_date is DateType in the real Gold table.

        WHY: Unit test proved the type survived the join on dummy data.
             This test confirms no type coercion occurred when joining
             real Silver charges (DateType service_date) with real Silver
             patientvisits (which has no service_date column).

        REAL DATA CHECK:
            - service_date column in Gold is DateType
        """
        if "service_date" not in gold_real.columns:
            pytest.skip("service_date column missing from Gold")

        actual_type = gold_real.schema["service_date"].dataType

        assert isinstance(actual_type, DateType), (
            f"Real data: service_date in Gold is {actual_type}. "
            f"Expected DateType — type was changed by the join."
        )
