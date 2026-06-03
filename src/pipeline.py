from typing import Dict
from pyspark.sql import DataFrame, SparkSession
from src.bronze.ingest import ingest_charges, ingest_patientvisits
from src.silver.transform_charges import build_silver_charges
from src.silver.transform_patientvisits import build_silver_patientvisits
from src.gold.aggregate import build_rcm_summary


def run_pipeline(bronze_charges: DataFrame,
                 bronze_patientvisits: DataFrame) -> Dict[str, DataFrame]:
    sc  = build_silver_charges(bronze_charges)
    spv = build_silver_patientvisits(bronze_patientvisits)
    return {
        "silver_charges":       sc,
        "silver_patientvisits": spv,
        "rcm_summary":          build_rcm_summary(sc, spv),
    }
