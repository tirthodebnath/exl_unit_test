"""
Silver — PatientVisits.
is_inpatient column removed (is_inpatient helper removed).
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.common.utils import computed_length_of_stay, has_insurance_balance, dedupe_latest


def cast_patientvisits_types(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("patient_admission_date",
                      F.to_date("patient_admission_date", "yyyy-MM-dd"))
          .withColumn("patient_discharge_date",
                      F.to_date("patient_discharge_date", "yyyy-MM-dd"))
          .withColumn("patient_final_bill_date",
                      F.to_date("patient_final_bill_date", "yyyy-MM-dd"))
          .withColumn("total_charges",
                      F.col("total_charges").cast("decimal(14,2)"))
          .withColumn("total_payments",
                      F.col("total_payments").cast("decimal(14,2)"))
          .withColumn("insurance_balance",
                      F.col("insurance_balance").cast("decimal(14,2)"))
          .withColumn("patient_balance",
                      F.col("patient_balance").cast("decimal(14,2)"))
          .withColumn("length_of_stay",
                      F.col("length_of_stay").cast("int"))
          .withColumn("expected_reimbursement_amount",
                      F.col("expected_reimbursement_amount").cast("decimal(14,2)"))
    )


def filter_invalid_patientvisits(df: DataFrame) -> DataFrame:
    return df.filter(
        F.col("rcm_client_id").isNotNull()
        & F.col("patient_account_number").isNotNull()
    )


def enrich_patientvisits(df: DataFrame) -> DataFrame:
    """Adds computed_los_days, has_insurance_balance, is_same_day_discharge."""
    with_los = df.withColumn(
        "computed_los_days",
        computed_length_of_stay(
            F.col("patient_admission_date"),
            F.col("patient_discharge_date"),
        )
    )
    return (
        with_los
        .withColumn("has_insurance_balance",
                    has_insurance_balance(F.col("insurance_balance")))
        .withColumn("is_same_day_discharge",
                    F.col("computed_los_days") == 0)
    )


def build_silver_patientvisits(bronze_patientvisits: DataFrame) -> DataFrame:
    typed    = cast_patientvisits_types(bronze_patientvisits)
    valid    = filter_invalid_patientvisits(typed)
    enriched = enrich_patientvisits(valid)
    return dedupe_latest(enriched,
                         key_cols=["rcm_client_id", "patient_account_number"],
                         order_col="_ingestion_timestamp")
