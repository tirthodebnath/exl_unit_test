"""
test_silver_charges.py
======================
Unit tests for the Silver charges transformation layer.

PURPOSE:
    These tests run on DUMMY DATA before the real pipeline executes.
    They verify that each Silver transformation step behaves correctly
    in isolation and when composed together.

SCOPE:
    Charges table only. Silver patientvisits tests were removed from
    this suite per project decision — they would duplicate the same
    patterns with different column names.

DATA SOURCE:
    All data is created in memory using make_charge() from conftest.py.
    No CSV files, no Volume, no Delta tables are touched.
"""

from datetime import datetime

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

from src.silver.transform_charges import build_silver_charges, cast_charges_types


# ---------------------------------------------------------------------------
# TestSilverServiceDate
# Tests that service_date is converted to the correct type in Silver
# ---------------------------------------------------------------------------
class TestSilverServiceDate:

    def test_service_date_is_datetime(self, spark, make_charge,
                                      bronze_charges_schema):
        """
        WHAT: After Silver processing, service_date must be DateType not StringType.

        WHY: RCM calculations depend on date arithmetic — claim age in days,
             AR aging buckets (30/60/90 day), late charge detection. These all
             require a real DateType. If service_date is still a string, every
             date calculation will fail or return wrong results.

        HOW:
            - Build one dummy charge with a valid service date string
            - Run it through the full Silver pipeline
            - Assert the output schema has DateType for service_date

        VALIDATES:
            - service_date column type is DateType in Silver output
        """
        # Build a single dummy charge row with a valid date string
        row = make_charge(
            service_date="2024-01-15",
            charge_posting_date="2024-01-16"   # must be >= service_date
        )
        df  = spark.createDataFrame([row], schema=bronze_charges_schema)

        # Run through the full Silver pipeline (cast → filter → enrich → dedupe)
        out = build_silver_charges(df)

        # service_date must now be DateType — not a string
        assert isinstance(out.schema["service_date"].dataType, DateType), (
            f"service_date must be DateType in Silver, "
            f"got: {out.schema['service_date'].dataType}"
        )


# ---------------------------------------------------------------------------
# TestSilverBusinessRules
# Tests that Silver enforces RCM business rules by filtering invalid records
# ---------------------------------------------------------------------------
class TestSilverBusinessRules:

    def test_post_date_not_before_service_date(self, spark, make_charge,
                                                bronze_charges_schema):
        """
        WHAT: Verify that Silver drops any charge where charge_posting_date
              is earlier than service_date.

        WHY: A charge cannot be posted before the service was rendered.
             If posting_date < service_date, it is a data entry error in
             the source system. Allowing this into Silver would corrupt:
               - AR aging calculations (claim appears older than it is)
               - Revenue reporting (revenue counted in wrong period)
               - Audit trails (impossible timeline)
             Silver is the right place to enforce this — Bronze preserves
             everything, Silver enforces business rules.

        HOW:
            - Create one valid charge (posting Jan 16, service Jan 15 — OK)
            - Create one invalid charge (posting Jan 14, service Jan 15 — WRONG)
            - Run both through build_silver_charges()
            - Assert no rows survive where posting < service

        VALIDATES:
            - Records with posting_date < service_date are filtered out
            - Valid records are not affected
        """
        # VALID: charge posted the day after service — correct RCM workflow
        valid = make_charge(
            charge_id="CHG001",
            service_date="2024-01-15",
            charge_posting_date="2024-01-16"    # posting AFTER service ✓
        )

        # INVALID: charge posted before it was rendered — data error
        invalid = make_charge(
            charge_id="CHG999",
            service_date="2024-01-15",
            charge_posting_date="2024-01-14"    # posting BEFORE service ✗
        )

        df  = spark.createDataFrame([valid, invalid], schema=bronze_charges_schema)
        out = build_silver_charges(df)

        # No row should survive where posting is before service
        bad_rows = out.filter(
            F.col("charge_posting_date") < F.col("service_date")
        ).count()

        assert bad_rows == 0, (
            f"{bad_rows} row(s) with charge_posting_date < service_date "
            f"survived Silver — business rule not enforced"
        )

        # The valid charge must still be present — not collateral damage
        assert out.count() == 1, (
            f"Expected 1 valid row in Silver output, got {out.count()}"
        )


# ---------------------------------------------------------------------------
# TestSilverDeduplication
# Tests that Silver deduplication keeps the latest version of each charge
# ---------------------------------------------------------------------------
class TestSilverDeduplication:

    def test_no_exact_duplicate_rows(self, spark, make_charge,
                                     bronze_charges_schema):
        """
        WHAT: When the same charge_id arrives twice (corrected resubmission),
              Silver must keep only the latest version. charge_id must be
              unique in the Silver output.

        WHY: Source RCM systems regularly resend corrected charge files.
             Without deduplication:
               - The same charge would be counted twice in Gold reports
               - Total billed amounts would be overstated
               - Denial rates would be wrong
             The dedupe_latest() function uses _ingestion_timestamp to
             determine which version is newer.

        HOW:
            - Create two rows with the same charge_id
            - Older version: ingested Monday, amount £100
            - Newer version: ingested Tuesday, amount £999 (correction)
            - Run through build_silver_charges()
            - Assert only 1 row survives and it is the Tuesday version

        VALIDATES:
            - Total rows == distinct charge_ids (no duplicates)
            - Latest version (Tuesday £999) is the one kept
        """
        # OLDER version — arrived Monday, original amount
        older = make_charge(
            charge_id="CHG001",
            amount="100.00",
            **{"_ingestion_timestamp": datetime(2024, 1, 15, 8, 0, 0)}
        )
        # NEWER version — arrived Tuesday with corrected amount
        newer = make_charge(
            charge_id="CHG001",
            amount="999.00",
            **{"_ingestion_timestamp": datetime(2024, 1, 16, 8, 0, 0)}
        )

        df  = spark.createDataFrame([older, newer], schema=bronze_charges_schema)
        out = build_silver_charges(df)

        # After dedup: total rows must equal distinct charge_ids
        total_rows   = out.count()
        distinct_ids = out.select("charge_id").distinct().count()

        assert total_rows == distinct_ids, (
            f"Duplicate charge_ids exist in Silver: "
            f"{total_rows} rows but only {distinct_ids} distinct charge_ids"
        )

        # Only 1 row should survive (the latest version)
        assert total_rows == 1, (
            f"Expected 1 row after dedup, got {total_rows}"
        )
