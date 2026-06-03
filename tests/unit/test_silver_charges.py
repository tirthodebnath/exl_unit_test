"""
Silver charges unit tests — charges table only.
"""
from datetime import datetime

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

from src.silver.transform_charges import build_silver_charges, cast_charges_types


class TestSilverServiceDate:

    def test_service_date_is_datetime(self, spark, make_charge,
                                      bronze_charges_schema):
        """
        After Silver processing service_date must be DateType, not a string.
        RCM date arithmetic (claim age, AR aging buckets) requires a real date.
        """
        row = make_charge(service_date="2024-01-15",
                          charge_posting_date="2024-01-16")
        df  = spark.createDataFrame([row], schema=bronze_charges_schema)
        out = build_silver_charges(df)

        assert isinstance(out.schema["service_date"].dataType, DateType), (
            "service_date must be DateType in Silver — "
            f"got {out.schema['service_date'].dataType}"
        )


class TestSilverBusinessRules:

    def test_post_date_not_before_service_date(self, spark, make_charge,
                                                bronze_charges_schema):
        """
        charge_posting_date must never be earlier than service_date in Silver.
        A charge posted before it was even rendered is a data error — Silver
        filters it out so Gold totals are never polluted by impossible dates.
        """
        valid   = make_charge(charge_id="CHG001",
                              service_date="2024-01-15",
                              charge_posting_date="2024-01-16")   # posting AFTER service ✓
        invalid = make_charge(charge_id="CHG999",
                              service_date="2024-01-15",
                              charge_posting_date="2024-01-14")   # posting BEFORE service ✗

        df  = spark.createDataFrame([valid, invalid], schema=bronze_charges_schema)
        out = build_silver_charges(df)

        # No row where posting is before service should survive Silver
        bad_rows = out.filter(
            F.col("charge_posting_date") < F.col("service_date")
        ).count()

        assert bad_rows == 0, (
            f"{bad_rows} row(s) with charge_posting_date < service_date survived Silver"
        )
        # The valid charge must still be present
        assert out.count() == 1


class TestSilverDeduplication:

    def test_no_exact_duplicate_rows(self, spark, make_charge,
                                     bronze_charges_schema):
        """
        After Silver dedup, charge_id must be unique.
        The same charge_id arriving twice (corrected resubmission) must
        produce exactly one row — the latest ingestion wins.
        """
        older = make_charge(
            charge_id="CHG001", amount="100.00",
            **{"_ingestion_timestamp": datetime(2024, 1, 15, 8, 0, 0)}
        )
        newer = make_charge(
            charge_id="CHG001", amount="999.00",
            **{"_ingestion_timestamp": datetime(2024, 1, 16, 8, 0, 0)}
        )

        df  = spark.createDataFrame([older, newer], schema=bronze_charges_schema)
        out = build_silver_charges(df)

        total    = out.count()
        distinct = out.select("charge_id").distinct().count()

        assert total == distinct, (
            f"Duplicate charge_ids found in Silver: {total} rows, "
            f"only {distinct} distinct charge_ids"
        )
        assert total == 1
