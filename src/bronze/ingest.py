from pyspark.sql import DataFrame, SparkSession
from src.common.schemas import CHARGES_BRONZE_SCHEMA, PATIENTVISITS_BRONZE_SCHEMA
from src.common.utils import with_audit_columns


def ingest_charges(spark: SparkSession, source_path: str,
                   source_format: str = "csv") -> DataFrame:
    reader = spark.read.format(source_format).schema(CHARGES_BRONZE_SCHEMA)
    if source_format == "csv":
        reader = reader.option("header", "true")
    return with_audit_columns(reader.load(source_path), source_file=source_path)


def ingest_patientvisits(spark: SparkSession, source_path: str,
                         source_format: str = "csv") -> DataFrame:
    reader = spark.read.format(source_format).schema(PATIENTVISITS_BRONZE_SCHEMA)
    if source_format == "csv":
        reader = reader.option("header", "true")
    return with_audit_columns(reader.load(source_path), source_file=source_path)
