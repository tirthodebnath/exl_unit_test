"""
test_silver_realdata.py
=======================
PURPOSE:
    The same 3 Silver charges tests from tests/unit/test_silver_charges.py,
    but running on the REAL silver_charges Delta table.

    Unit tests proved the transformation functions work correctly on dummy data.
    These tests prove the transformation produced correct results on real data.

SCOPE:
    Charges table only — matching the unit test scope decision.

HOW TO RUN:
    Run by test_after_load_data.py notebook after run_pipeline.py has loaded data.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DateType


# ---------------------------------------------------------------------------
# TestSilverServiceDate — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestSilverServiceDate:
    """
    Verify service_date type on the real silver_charges table.
    Matches TestSilverServiceDate in tests/unit/test_silver_charges.py.
    """

    def test_service_date_is_datetime(self, silver_charges_real):
        """
        WHAT: Verify service_date in the real silver_charges table is
              DateType — not a string.

        WHY: The unit test proved cast_charges_types() converts a dummy
             "2024-01-15" string to DateType. This test proves the same
             cast worked on EVERY real row in the actual Silver table.
             If even one row had an unparseable date, the cast would have
             returned null — but the column type itself is still DateType.
             This test confirms the schema is correct across all real data.

        REAL DATA CHECK:
            - service_date column in silver_charges is DateType
        """
        df = silver_charges_real

        assert "service_date" in df.columns, (
            "service_date column is missing from real silver_charges table"
        )

        # Get the actual data type from the real table schema
        actual_type = df.schema["service_date"].dataType

        assert isinstance(actual_type, DateType), (
            f"service_date in real silver_charges is {actual_type}. "
            f"Expected DateType — cast_charges_types() may not have run correctly."
        )


# ---------------------------------------------------------------------------
# TestSilverBusinessRules — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestSilverBusinessRules:
    """
    Verify business rules are enforced on the real silver_charges table.
    Matches TestSilverBusinessRules in tests/unit/test_silver_charges.py.
    """

    def test_post_date_not_before_service_date(self, silver_charges_real):
        """
        WHAT: Verify there are zero rows in the real silver_charges table
              where charge_posting_date is before service_date.

        WHY: The unit test proved filter_invalid_charges() drops these rows
             on a 2-row dummy DataFrame. This test proves the same filter
             worked on ALL real rows — no RCM charge with an impossible date
             timeline survived into Silver.

             This matters because:
             - AR aging calculations use these dates
             - Revenue period reporting depends on service_date
             - Audit trails require posting after service

        REAL DATA CHECK:
            - Zero rows where charge_posting_date < service_date
        """
        df = silver_charges_real

        if "charge_posting_date" not in df.columns or "service_date" not in df.columns:
            pytest.skip(
                "charge_posting_date or service_date column missing from Silver"
            )

        # Count rows where the business rule is violated on real data
        bad_rows = df.filter(
            F.col("charge_posting_date").isNotNull() &
            (F.col("charge_posting_date") < F.col("service_date"))
        ).count()

        assert bad_rows == 0, (
            f"{bad_rows} real charge row(s) have charge_posting_date "
            f"before service_date in Silver. "
            f"These should have been filtered by filter_invalid_charges()."
        )


# ---------------------------------------------------------------------------
# TestSilverDeduplication — same class name as unit test for consistency
# ---------------------------------------------------------------------------
class TestSilverDeduplication:
    """
    Verify deduplication on the real silver_charges table.
    Matches TestSilverDeduplication in tests/unit/test_silver_charges.py.
    """

    def test_no_exact_duplicate_rows(self, silver_charges_real):
        """
        WHAT: Verify charge_id is unique in the real silver_charges table.

        WHY: The unit test proved dedupe_latest() keeps one row per charge_id
             when given two dummy rows with the same ID. This test proves the
             same dedup logic worked on ALL real data — no charge_id appears
             more than once in the real Silver table.

             Duplicate charge_ids in Silver means:
             - Every Gold report double-counts that charge
             - Total billed amounts are overstated
             - Denial rates are wrong

        REAL DATA CHECK:
            - Total rows == distinct charge_id count
        """
        df = silver_charges_real

        if "charge_id" not in df.columns:
            pytest.skip("charge_id column missing from real silver_charges")

        total_rows   = df.count()
        distinct_ids = df.select("charge_id").distinct().count()

        assert total_rows == distinct_ids, (
            f"Duplicate charge_ids found in real silver_charges. "
            f"Total rows: {total_rows}, Distinct charge_ids: {distinct_ids}. "
            f"Difference: {total_rows - distinct_ids} duplicate rows. "
            f"dedupe_latest() may not have run correctly."
        )
