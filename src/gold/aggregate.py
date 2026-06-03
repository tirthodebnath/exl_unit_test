from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_rcm_summary(silver_charges: DataFrame,
                      silver_patientvisits: DataFrame) -> DataFrame:
    """LEFT JOIN charges → patientvisits on rcm_client_id (one-to-many)."""
    visit_cols = [c for c in silver_patientvisits.columns if c != "rcm_client_id"]
    pv = silver_patientvisits.select(
        F.col("rcm_client_id"),
        *[F.col(c).alias(f"pv_{c}") for c in visit_cols]
    )
    return silver_charges.join(pv, on="rcm_client_id", how="left")
