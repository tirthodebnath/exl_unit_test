"""
tests/integration/gold/test_ogom_charges.py
============================================
PURPOSE:
    Data quality tests for the real workspace.tirtho_db.gold_ogom_charges
    Delta table.

    These tests run AFTER run_pipeline.py has created gold_ogom_charges.
    They verify that:
        1. The join conditions were correctly applied on real data
        2. All computed columns (charge_age, late_charge_flag,
           charge_lag_days, charge_capture_days) are correct
        3. No charges were lost or invented during the join
        4. Business rules from the original SQL hold on real data

    Run by: notebooks/run_tests_gold.py

SQL THIS TABLE MAPS TO:
    prod_lca_unrestricted.bassett_epic_acute_gold.ogomcharges
    (charges LEFT JOIN patientvisits with 3 conditions + discharge filter)

COLUMNS TESTED:
    Identity:    charge_id, rcm_client_id
    Amounts:     charge_amount
    Computed:    charge_age, ogom_transaction_type, late_charge_flag,
                 charge_lag_days, charge_capture_days
    Join proof:  discharge_date, pv columns, LEFT JOIN guarantee
"""

import pytest
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Helper: load the real gold_ogom_charges table
# ---------------------------------------------------------------------------
def _get_ogom_charges(spark):
    """
    Load the real gold_ogom_charges Delta table.

    WHY A HELPER:
        All tests in this file need the same DataFrame. Centralising the
        table load means if the table name changes, we change it in one
        place only. Also skips cleanly with a clear message if the table
        does not exist yet.

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
            f"Run run_pipeline.py or create_gold_v2.py first. Error: {e}"
        )


# ---------------------------------------------------------------------------
# TestOGOMChargesIdentity — critical columns must never be null
# ---------------------------------------------------------------------------
class TestOGOMChargesIdentity:
    """
    Identity checks — the columns that make each row traceable.
    A row without charge_id or rcm_client_id cannot be used in any report.
    """

    def test_charge_id_not_null(self, spark):
        """
        WHAT: charge_id must never be null in gold_ogom_charges.

        WHY: charge_id is the primary identifier of every charge line.
             The Silver filter already removed null charge_ids.
             If any null survives to Gold, that row is untraceable —
             it cannot be linked to a patient, a claim, or a payer.

        REAL DATA CHECK:
            Zero rows where charge_id is null.
        """
        df       = _get_ogom_charges(spark)
        null_cnt = df.filter(F.col("charge_id").isNull()).count()

        assert null_cnt == 0, (
            f"{null_cnt} rows have null charge_id in gold_ogom_charges. "
            f"Silver filter should have removed these before Gold."
        )

    def test_rcm_client_id_not_null(self, spark):
        """
        WHAT: rcm_client_id must never be null in gold_ogom_charges.

        WHY: rcm_client_id is the join key — it identifies which hospital
             client the charge belongs to. A null rcm_client_id means the
             row cannot be filtered by client, making it invisible to any
             client-specific report.

        REAL DATA CHECK:
            Zero rows where rcm_client_id is null.
        """
        df       = _get_ogom_charges(spark)
        null_cnt = df.filter(F.col("rcm_client_id").isNull()).count()

        assert null_cnt == 0, (
            f"{null_cnt} rows have null rcm_client_id in gold_ogom_charges."
        )


# ---------------------------------------------------------------------------
# TestOGOMChargesAmounts — financial integrity
# ---------------------------------------------------------------------------
class TestOGOMChargesAmounts:
    """
    Financial integrity checks on charge_amount.
    Incorrect amounts corrupt all revenue reporting.
    """

    def test_charge_amount_not_negative(self, spark):
        """
        WHAT: charge_amount must never be negative in gold_ogom_charges.

        WHY: Silver filter removed negative amounts from silver_charges.
             If any negative amount appears in Gold, something added it
             after Silver — which should be impossible. Negative revenue
             figures would distort all Gold financial reports.

        REAL DATA CHECK:
            Zero rows where charge_amount < 0.
        """
        df      = _get_ogom_charges(spark)
        neg_cnt = df.filter(F.col("charge_amount") < 0).count()

        assert neg_cnt == 0, (
            f"{neg_cnt} rows have negative charge_amount in gold_ogom_charges."
        )

    def test_charge_amount_not_null(self, spark):
        """
        WHAT: charge_amount must never be null in gold_ogom_charges.

        WHY: A charge with no amount has no financial value.
             The Silver filter removed null amounts. If any null appears
             in Gold, the filter did not run correctly and revenue
             totals will be understated.

        REAL DATA CHECK:
            Zero rows where charge_amount is null.
        """
        df       = _get_ogom_charges(spark)
        null_cnt = df.filter(F.col("charge_amount").isNull()).count()

        assert null_cnt == 0, (
            f"{null_cnt} rows have null charge_amount in gold_ogom_charges."
        )


# ---------------------------------------------------------------------------
# TestOGOMChargesComputedColumns — verify SQL-derived logic on real data
# ---------------------------------------------------------------------------
class TestOGOMChargesComputedColumns:
    """
    Verifies the 4 computed columns are correct on real data.
    These columns are derived from the join output and do not come
    from Silver directly.
    """

    def test_ogom_transaction_type_always_charge(self, spark):
        """
        WHAT: ogom_transaction_type must be "Charge" on every single row.

        WHY: Maps to the literal "Charge" AS OGOMTransactionType in the SQL.
             This column identifies all rows in this table as charge
             transactions. If any row has a different value, the literal
             computation failed.

        REAL DATA CHECK:
            All distinct ogom_transaction_type values == {"Charge"}
        """
        df             = _get_ogom_charges(spark)
        distinct_types = {
            r["ogom_transaction_type"]
            for r in df.select("ogom_transaction_type").distinct().collect()
        }

        assert distinct_types == {"Charge"}, (
            f"ogom_transaction_type has unexpected values: {distinct_types}. "
            f"Every row must have exactly 'Charge'."
        )

    def test_charge_age_null_when_no_discharge(self, spark):
        """
        WHAT: charge_age must be null whenever discharge_date is null.

        WHY: Maps to:
                 CASE WHEN pv.PatientDischargeDate IS NULL THEN NULL
                      ELSE DATEDIFF(...) END AS ChargeAge
             A charge cannot have an age relative to a discharge that
             has not happened. If charge_age is not null when discharge_date
             is null, the CASE logic was computed incorrectly.

        REAL DATA CHECK:
            Zero rows where discharge_date is null AND charge_age is not null.
        """
        df = _get_ogom_charges(spark)

        # Rows where discharge is null must also have null charge_age
        violated = df.filter(
            F.col("discharge_date").isNull() &
            F.col("charge_age").isNotNull()
        ).count()

        assert violated == 0, (
            f"{violated} rows have null discharge_date but non-null charge_age. "
            f"charge_age must be null when no discharge date exists."
        )

    def test_charge_age_computed_when_discharged(self, spark):
        """
        WHAT: charge_age must not be null when discharge_date is present.

        WHY: When discharge_date is not null, charge_age must be computed
             as DATEDIFF(posting_date, discharge_date). A null charge_age
             with a valid discharge_date means the computation was skipped.

        REAL DATA CHECK:
            Zero rows where discharge_date is not null AND charge_age is null.
        """
        df = _get_ogom_charges(spark)

        violated = df.filter(
            F.col("discharge_date").isNotNull() &
            F.col("charge_age").isNull()
        ).count()

        assert violated == 0, (
            f"{violated} rows have a discharge_date but null charge_age. "
            f"charge_age must be computed when discharge_date is present."
        )

    def test_late_charge_flag_null_when_no_discharge(self, spark):
        """
        WHAT: late_charge_flag must be null when discharge_date is null.

        WHY: Maps to:
                 CASE WHEN pv.PatientDischargeDate IS NULL THEN NULL ...
             The flag can only be determined after discharge. If the flag
             is not null when there is no discharge, the CASE logic is wrong.

        REAL DATA CHECK:
            Zero rows where discharge_date is null AND late_charge_flag is not null.
        """
        df = _get_ogom_charges(spark)

        violated = df.filter(
            F.col("discharge_date").isNull() &
            F.col("late_charge_flag").isNotNull()
        ).count()

        assert violated == 0, (
            f"{violated} rows have null discharge_date but non-null late_charge_flag. "
            f"late_charge_flag must be null when no discharge date exists."
        )

    def test_late_charge_flag_values_are_valid(self, spark):
        """
        WHAT: late_charge_flag must only contain 0, 1, or null.

        WHY: The CASE expression produces only NULL, 0, or 1.
             Any other value means the computation produced unexpected output.

        REAL DATA CHECK:
            All distinct late_charge_flag values ⊆ {0, 1, None}
        """
        df = _get_ogom_charges(spark)

        invalid = df.filter(
            F.col("late_charge_flag").isNotNull() &
            ~F.col("late_charge_flag").isin(0, 1)
        ).count()

        assert invalid == 0, (
            f"{invalid} rows have late_charge_flag values other than 0, 1, or null."
        )

    def test_charge_lag_days_computed_when_dates_exist(self, spark):
        """
        WHAT: charge_lag_days must not be null when both charge_posting_date
              and service_date are present.

        WHY: Maps to:
                 CASE WHEN c.ChargePostingDate IS NOT NULL
                      AND c.ServiceDate IS NOT NULL
                      THEN DATEDIFF(c.ChargePostingDate, c.ServiceDate)
                 END AS ChargeLagDays
             If charge_lag_days is null when both dates exist, the
             DATEDIFF was not computed — we cannot measure charge lag.

        REAL DATA CHECK:
            Zero rows where both dates are present AND charge_lag_days is null.
        """
        df = _get_ogom_charges(spark)

        violated = df.filter(
            F.col("posting_date").isNotNull() &
            F.col("service_date").isNotNull() &
            F.col("charge_lag_days").isNull()
        ).count()

        assert violated == 0, (
            f"{violated} rows have posting_date AND service_date "
            f"but null charge_lag_days."
        )

    def test_charge_capture_days_computed_when_dates_exist(self, spark):
        """
        WHAT: charge_capture_days must not be null when both
              charge_posting_date and admit_date are present.

        WHY: Maps to:
                 CASE WHEN c.ChargePostingDate IS NOT NULL
                      AND pv.PatientAdmissionDate IS NOT NULL
                      THEN DATEDIFF(c.ChargePostingDate, pv.PatientAdmissionDate)
                      ELSE NULL
                 END AS ChargeCaptureDays
             A null capture_days when both dates exist means the
             computation was skipped — we lose visibility into capture lag.

        REAL DATA CHECK:
            Zero rows where both dates are present AND charge_capture_days is null.
        """
        df = _get_ogom_charges(spark)

        violated = df.filter(
            F.col("posting_date").isNotNull() &
            F.col("admit_date").isNotNull() &
            F.col("charge_capture_days").isNull()
        ).count()

        assert violated == 0, (
            f"{violated} rows have posting_date AND admit_date "
            f"but null charge_capture_days."
        )


# ---------------------------------------------------------------------------
# TestOGOMChargesJoinIntegrity — join conditions verified on real data
# ---------------------------------------------------------------------------
class TestOGOMChargesJoinIntegrity:
    """
    Verifies the LEFT JOIN conditions produced correct results on real data.
    These tests confirm that:
        - All charges survived (LEFT JOIN guarantee)
        - Only discharged patients were joined
        - No charges were lost or invented
    """

    def test_no_charges_dropped_from_silver(self, spark):
        """
        WHAT: Distinct charge_ids in gold_ogom_charges must equal the
              Silver charges row count.

        WHY: LEFT JOIN guarantees every charge appears in Gold.
             If any charge_id is missing, the join became an INNER JOIN
             somewhere — unreconciled charges would disappear from reports.

        REAL DATA CHECK:
            gold.select("charge_id").distinct().count() == silver_charges.count()
        """
        df           = _get_ogom_charges(spark)
        gold_distinct = df.select("charge_id").distinct().count()

        try:
            silver_count = spark.table(
                "workspace.tirtho_db.silver_charges"
            ).count()
        except Exception as e:
            pytest.skip(f"silver_charges not available for comparison. Error: {e}")

        assert gold_distinct == silver_count, (
            f"Gold OGOM has {gold_distinct} distinct charge_ids "
            f"but Silver has {silver_count} charges. "
            f"Difference: {abs(gold_distinct - silver_count)}"
        )

    def test_discharge_date_not_null_where_visit_joined(self, spark):
        """
        WHAT: Where a visit was matched (admit_date is not null),
              discharge_date must also not be null.

        WHY: The join condition includes AND pv.PatientDischargeDate IS NOT NULL.
             Only discharged patients are eligible to be matched.
             If discharge_date is null for a matched visit, the filter
             condition was not enforced on real data.

        REAL DATA CHECK:
            Zero rows where admit_date is not null AND discharge_date is null.
        """
        df = _get_ogom_charges(spark)

        violated = df.filter(
            F.col("admit_date").isNotNull() &
            F.col("discharge_date").isNull()
        ).count()

        assert violated == 0, (
            f"{violated} Gold OGOM rows have admit_date (visit joined) "
            f"but null discharge_date. "
            f"The discharge date IS NOT NULL join condition was not enforced."
        )

    def test_gold_has_at_least_one_row(self, spark):
        """
        WHAT: gold_ogom_charges must have at least 1 row.

        WHY: An empty Gold table means either Silver charges was empty
             or the join and compute logic failed completely. All
             downstream OGOM reports would show no data.

        REAL DATA CHECK:
            Row count >= 1.
        """
        df    = _get_ogom_charges(spark)
        count = df.count()

        assert count >= 1, (
            f"gold_ogom_charges is empty. "
            f"Run run_pipeline.py to populate it."
        )
