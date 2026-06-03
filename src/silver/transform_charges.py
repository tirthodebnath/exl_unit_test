"""
Silver — Charges.
Steps: cast → filter (incl. post_date >= service_date) → enrich → dedupe.
is_corrected_charge and is_late_charge removed (is_y_flag helper removed).
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.common.utils import classify_charge_amount, dedupe_latest


def cast_charges_types(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("service_date",          F.to_date("service_date", "yyyy-MM-dd"))
          .withColumn("charge_posting_date",   F.to_date("charge_posting_date", "yyyy-MM-dd"))
          .withColumn("amount",                F.col("amount").cast("decimal(12,2)"))
          .withColumn("detail_charge_quantity",F.col("detail_charge_quantity").cast("decimal(10,2)"))
          .withColumn("number_of_units",       F.col("number_of_units").cast("int"))
    )


def filter_invalid_charges(df: DataFrame) -> DataFrame:
    """
    Drop records that violate RCM invariants.
    Also enforces: charge_posting_date must not be before service_date.
    """
    return df.filter(
        F.col("charge_id").isNotNull()
        & F.col("rcm_client_id").isNotNull()
        & F.col("patient_account_number").isNotNull()
        & F.col("service_date").isNotNull()
        & F.col("amount").isNotNull()
        & (F.col("amount") >= 0)
        & (
            F.col("charge_posting_date").isNull() |
            (F.col("charge_posting_date") >= F.col("service_date"))
        )
    )


def enrich_charges(df: DataFrame) -> DataFrame:
    """Add amount_band. Corrected/late charge flags removed (is_y_flag removed)."""
    return df.withColumn("amount_band", classify_charge_amount(F.col("amount")))


def build_silver_charges(bronze_charges: DataFrame) -> DataFrame:
    typed    = cast_charges_types(bronze_charges)
    valid    = filter_invalid_charges(typed)
    enriched = enrich_charges(valid)
    return dedupe_latest(enriched, key_cols=["charge_id"],
                         order_col="_ingestion_timestamp")
