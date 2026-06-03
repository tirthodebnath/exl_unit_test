"""
Gold reconciliation tests.

Gold is a LEFT JOIN of Silver charges → Silver patientvisits on rcm_client_id.
Because the join is one-to-many, Gold has MORE rows than Silver charges
(one charge can fan out to multiple visit rows).

All reconciliation tests therefore work on DISTINCT charge_ids in Gold,
not raw row counts, so the fan-out does not cause false failures.
"""
import pytest
from decimal import Decimal
from pyspark.sql import functions as F

from src.silver.transform_charges import build_silver_charges
from src.silver.transform_patientvisits import build_silver_patientvisits
from src.gold.aggregate import build_rcm_summary


@pytest.fixture
def silver_and_gold(bronze_charges_df, bronze_patientvisits_df):
    sc   = build_silver_charges(bronze_charges_df)
    spv  = build_silver_patientvisits(bronze_patientvisits_df)
    gold = build_rcm_summary(sc, spv)
    return sc, gold


class TestGoldReconciliation:

    def test_charge_count_matches_silver(self, silver_and_gold):
        """
        The number of distinct charge_ids in Gold must equal the row count
        in Silver charges.  If a charge is silently dropped or duplicated
        at the join step, this test fails.
        """
        silver, gold = silver_and_gold
        silver_count      = silver.count()
        gold_distinct_cnt = gold.select("charge_id").distinct().count()

        assert gold_distinct_cnt == silver_count, (
            f"Silver has {silver_count} charges; "
            f"Gold has {gold_distinct_cnt} distinct charge_ids"
        )

    def test_total_charge_amount_matches_silver(self, silver_and_gold):
        """
        Sum of charge amounts for distinct charge_ids in Gold must equal the
        sum in Silver.  The one-to-many join fans out rows, so we dedupe on
        charge_id before summing — each charge must be counted exactly once.
        """
        silver, gold = silver_and_gold

        silver_total = silver.agg(F.sum("amount").alias("t")).first()["t"]
        gold_total   = (
            gold.dropDuplicates(["charge_id"])
                .agg(F.sum("amount").alias("t"))
                .first()["t"]
        )

        assert silver_total == gold_total, (
            f"Amount mismatch — Silver total: {silver_total}, "
            f"Gold total (deduped): {gold_total}"
        )

    def test_no_new_charge_ids_introduced_in_gold(self, silver_and_gold):
        """
        Every charge_id in Gold must also exist in Silver.
        The join must never invent new charge_ids.
        """
        silver, gold = silver_and_gold

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        new_ids = gold_ids - silver_ids
        assert len(new_ids) == 0, (
            f"Gold contains charge_ids not in Silver: {new_ids}"
        )

    def test_no_silver_charge_ids_dropped_in_gold(self, silver_and_gold):
        """
        Every charge_id in Silver must appear in Gold.
        A LEFT JOIN on charges guarantees this — this test catches if someone
        accidentally changes it to an INNER JOIN, which would silently drop
        charges whose rcm_client_id has no matching visit.
        """
        silver, gold = silver_and_gold

        silver_ids = {r["charge_id"]
                      for r in silver.select("charge_id").collect()}
        gold_ids   = {r["charge_id"]
                      for r in gold.select("charge_id").distinct().collect()}

        dropped_ids = silver_ids - gold_ids
        assert len(dropped_ids) == 0, (
            f"Silver charge_ids missing from Gold: {dropped_ids}"
        )
