"""
Shared pure helpers. is_y_flag and is_inpatient removed per spec.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column


def classify_charge_amount(amount_col: Column) -> Column:
    """LOW / MEDIUM / HIGH / JUMBO / UNKNOWN banding for charge amounts."""
    return (
        F.when(amount_col.isNull() | (amount_col <= 0), F.lit("UNKNOWN"))
         .when(amount_col <= 100,   F.lit("LOW"))
         .when(amount_col <= 1000,  F.lit("MEDIUM"))
         .when(amount_col <= 10000, F.lit("HIGH"))
         .otherwise(F.lit("JUMBO"))
    )


def computed_length_of_stay(admission_col: Column, discharge_col: Column) -> Column:
    """Length of stay in days. Returns null if either date is null."""
    return F.datediff(discharge_col, admission_col)


def has_insurance_balance(balance_col: Column) -> Column:
    """True when insurance_balance is positive."""
    return balance_col.isNotNull() & (balance_col > 0)


def with_audit_columns(df: DataFrame, source_file: str) -> DataFrame:
    """Stamp _ingestion_timestamp and _source_file on every Bronze row."""
    return (
        df.withColumn("_ingestion_timestamp", F.current_timestamp())
          .withColumn("_source_file", F.lit(source_file))
    )


def dedupe_latest(df: DataFrame, key_cols: list, order_col: str) -> DataFrame:
    """Keep the most recent record per key based on order_col descending."""
    from pyspark.sql.window import Window
    w = Window.partitionBy(*key_cols).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_rn", F.row_number().over(w))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )
