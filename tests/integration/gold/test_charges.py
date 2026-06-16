"""
tests/integration/gold/test_charges.py
========================================
PURPOSE:
    Charge-focused data quality tests on the real gold_rcm_summary_v2 table.

    These tests verify that charge data is preserved correctly through the
    enhanced Gold join (3 conditions + discharge date filter). They answer
    the question: did the charges survive the join intact?

GOLD V2 JOIN (for reference):
    silver_charges LEFT JOIN silver_patientvisits
        ON  rcm_client_id = rcm_client_id
        AND rcm_npi = rcm_npi
        AND patient_account_number = patient_account_number
        AND patient_discharge_date IS NOT NULL

TESTS IN THIS FILE (charge-focused):
    TestGoldChargeReconciliation
        1. test_charge_count_matches_silver
        2. test_total_charge_amount_matches_silver
        3. test_no_new_charge_ids_in_gold
        4. test_no_silver_charge_ids_dropped
        5. test_amount_band_valid_in_gold
        6. test_service_date_is_date_type_in_gold
"""

import pytest
from decimal import Decimal
from pyspark.sql import functions as F
from pyspark.sql.types import DateType


class TestGoldChargeReconciliation:
    """
    Reconciliation tests confirm no charge data was lost, invented,
    or corrupted when Silver charges were joined to Silver patientvisits.

    All tests use DISTINCT charge_ids in Gold to handle the case where
    one charge matches multiple visit rows (though V2 join is more precise).
    """

    def test_charge_count_matches_silver(self, silver_and_gold_v2):
        """
        WHAT: Distinct charge_ids in Gold V2 must equal Silver charges row count.

        WHY: Every charge that exists in Silver must appear exactly once
             in Gold (LEFT JOIN guarantees presence). If the count differs,
             charges were either dropped or invented at the join step —
             both are critical defects that break revenue reporting.

        REAL DATA CHECK:
            gold.select("charge_id").distinct().count() == silver.count()
        """
        silver, gold = silver_and_gold_v2

        silver_count      = silver.count()
        gold_distinct_cnt = gold.select("charge_id").distinct().count()

        assert gold_distinct_cnt == silver_count, (
            f"Gold V2 has {gold_distinct_cnt} distinct charge_ids but "
            f"Silver has {silver_count} charges. "
            f"Difference: {abs(gold_distinct_cnt - silver_count)}"
        )

    def test_total_charge_amount_matches_silver(self, silver_and_gold_v2):
        """
        WHAT: Sum of charge amounts in Gold V2 (deduped on charge_id) must
              equal the total in Silver charges.

        WHY: If any charge amount was altered or lost in the join, revenue
             totals in Gold reports would be wrong. Deduping on charge_id
             before summing ensures we count each charge exactly once —
             the V2 join may still produce multiple rows per charge if
             multiple visits match all 3 conditions.

        REAL DATA CHECK:
            gold.dropDuplicates(["charge_id"]).sum(amount) == silver.sum(amount)
        """
        silver, gold = silver_and_gold_v2

        silver_total = silver.agg(F.sum("amount").alias("t")).first()["t"]
        gold_total   = (
            gold.dropDuplicates(["charge_id"])
                .agg(F.sum("amount").alias("t"))
                .first()["t"]
        )

        assert silver_total == gold_total, (
            f"Charge amount mismatch between Silver and Gold V2. "
            f"Silver total: {silver_total}, Gold total (deduped): {gold_total}"
        )

    def test_no_new_charge_ids_in_gold(self, silver_and_gold_v2):
        """
        WHAT: Every charge_id in Gold V2 must also exist in Silver.
              The join must never invent charge_ids that do not exist in Silver.

        WHY: A phantom charge_id in Gold means revenue data was created
             from nowhere. This would cause Gold reports to show more charges
             than actually exist — a fundamental data integrity failure.

        REAL DATA CHECK:
            set(gold charge_ids) - set(silver charge_ids) == empty set
        """
        silver, gold = silver_and_gold_v2

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        phantom = gold_ids - silver_ids
        assert len(phantom) == 0, (
            f"Gold V2 contains {len(phantom)} charge_id(s) not in Silver: "
            f"{phantom}"
        )

    def test_no_silver_charge_ids_dropped(self, silver_and_gold_v2):
        """
        WHAT: Every charge_id in Silver must appear in Gold V2.
              No charge should be silently dropped by the join.

        WHY: This is the most critical Gold test. The LEFT JOIN guarantees
             that all charges appear in the output regardless of whether
             a matching visit exists. If a charge is missing from Gold,
             it means the join accidentally became an INNER JOIN somewhere —
             unreconciled charges would be invisible to billing teams.

        REAL DATA CHECK:
            set(silver charge_ids) - set(gold charge_ids) == empty set
        """
        silver, gold = silver_and_gold_v2

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        dropped = silver_ids - gold_ids
        assert len(dropped) == 0, (
            f"{len(dropped)} Silver charge_id(s) missing from Gold V2. "
            f"LEFT JOIN may have become INNER JOIN: {dropped}"
        )

    def test_amount_band_valid_in_gold(self, gold_v2_real):
        """
        WHAT: amount_band in Gold V2 must only contain the 5 valid band values.

        WHY: amount_band is computed in Silver and must survive the Gold join
             unchanged. An invalid band value means the join corrupted a
             derived column, which would break any downstream filter on band.

        REAL DATA CHECK:
            All distinct amount_band values ⊆ {LOW, MEDIUM, HIGH, JUMBO, UNKNOWN}
        """
        if "amount_band" not in gold_v2_real.columns:
            pytest.skip("amount_band column missing from Gold V2")

        allowed_bands = {"LOW", "MEDIUM", "HIGH", "JUMBO", "UNKNOWN"}
        actual_bands  = {
            r["amount_band"]
            for r in gold_v2_real.select("amount_band").distinct().collect()
        }
        unexpected = actual_bands - allowed_bands

        assert len(unexpected) == 0, (
            f"Gold V2 contains invalid amount_band values: {unexpected}. "
            f"Allowed values: {allowed_bands}"
        )

    def test_service_date_is_date_type_in_gold(self, gold_v2_real):
        """
        WHAT: service_date must remain DateType in Gold V2 after the join.

        WHY: service_date is cast to DateType in Silver. Spark joins can
             occasionally cause type coercion. If service_date reverts to
             StringType in Gold, all date-based RCM calculations (AR aging,
             claim timelines) would silently return wrong results.

        REAL DATA CHECK:
            gold_v2.schema["service_date"].dataType is DateType
        """
        if "service_date" not in gold_v2_real.columns:
            pytest.skip("service_date column missing from Gold V2")

        actual_type = gold_v2_real.schema["service_date"].dataType

        assert isinstance(actual_type, DateType), (
            f"service_date in Gold V2 is {actual_type}. "
            f"Expected DateType — type was changed by the join."
        )
