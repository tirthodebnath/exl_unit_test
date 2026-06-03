# Databricks notebook source
# MAGIC %pip install chispa==0.10.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "tirtho_db"
VOLUME  = "tirtho_uploaded_files"

CHARGES_LANDING       = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/charges.csv"
PATIENTVISITS_LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/patientvisits.csv"

BRONZE_CHARGES_TABLE        = f"{CATALOG}.{SCHEMA}.bronze_charges"
BRONZE_PATIENTVISITS_TABLE  = f"{CATALOG}.{SCHEMA}.bronze_patientvisits"
SILVER_CHARGES_TABLE        = f"{CATALOG}.{SCHEMA}.silver_charges"
SILVER_PATIENTVISITS_TABLE  = f"{CATALOG}.{SCHEMA}.silver_patientvisits"
GOLD_RCM_TABLE              = f"{CATALOG}.{SCHEMA}.gold_rcm_summary"

print(f"Charges landing:       {CHARGES_LANDING}")
print(f"PatientVisits landing: {PATIENTVISITS_LANDING}")

# COMMAND ----------

# MAGIC %md ## Setup — copy repo to writable path and import src

# COMMAND ----------

import os, sys, shutil

NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils()
           .notebook().getContext().notebookPath().get()
)
REPO_PATH = "/Workspace" + os.path.dirname(NOTEBOOK_PATH).rsplit("/notebooks", 1)[0]
WORK_DIR  = "/tmp/exl_pipeline_work"

# Remove only the specific work subdirectory, not /tmp/ itself
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

shutil.copytree(REPO_PATH, WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
sys.dont_write_bytecode = True

print(f"Repo root: {REPO_PATH}")
print(f"Work dir:  {WORK_DIR}")

# Sanity check
import src.common.schemas
print(f"src found: {src.common.schemas.__file__}")

# COMMAND ----------

from pyspark.sql import SparkSession
from src.bronze.ingest import ingest_charges, ingest_patientvisits
from src.silver.transform_charges import build_silver_charges
from src.silver.transform_patientvisits import build_silver_patientvisits
from src.gold.aggregate import build_rcm_summary

spark = SparkSession.builder.getOrCreate()
print("Imports OK")

# COMMAND ----------

# MAGIC %md ## Bronze — raw ingest from Volume

# COMMAND ----------

bronze_charges       = ingest_charges(spark, CHARGES_LANDING)
bronze_patientvisits = ingest_patientvisits(spark, PATIENTVISITS_LANDING)

print(f"Bronze charges:       {bronze_charges.count()} rows")
print(f"Bronze patientvisits: {bronze_patientvisits.count()} rows")

bronze_charges.write.format("delta").mode("overwrite") \
    .saveAsTable(BRONZE_CHARGES_TABLE)
bronze_patientvisits.write.format("delta").mode("overwrite") \
    .saveAsTable(BRONZE_PATIENTVISITS_TABLE)

print("Bronze tables written.")

# COMMAND ----------

# MAGIC %md ## Silver — cast, filter, enrich, dedupe

# COMMAND ----------

silver_charges       = build_silver_charges(spark.table(BRONZE_CHARGES_TABLE))
silver_patientvisits = build_silver_patientvisits(spark.table(BRONZE_PATIENTVISITS_TABLE))

print(f"Silver charges:       {silver_charges.count()} rows")
print(f"Silver patientvisits: {silver_patientvisits.count()} rows")

silver_charges.write.format("delta").mode("overwrite") \
    .saveAsTable(SILVER_CHARGES_TABLE)
silver_patientvisits.write.format("delta").mode("overwrite") \
    .saveAsTable(SILVER_PATIENTVISITS_TABLE)

print("Silver tables written.")

# COMMAND ----------

# MAGIC %md ## Gold — LEFT JOIN charges → patientvisits on rcm_client_id

# COMMAND ----------

gold = build_rcm_summary(
    spark.table(SILVER_CHARGES_TABLE),
    spark.table(SILVER_PATIENTVISITS_TABLE),
)

print(f"Gold RCM summary: {gold.count()} rows")

gold.write.format("delta").mode("overwrite") \
    .saveAsTable(GOLD_RCM_TABLE)

print("Gold table written.")

# COMMAND ----------

# MAGIC %md ## Preview Gold output

# COMMAND ----------

spark.table(GOLD_RCM_TABLE).select(
    "charge_id",
    "rcm_client_id",
    "amount",
    "amount_band",
    "service_date",
    "charge_posting_date",
    "pv_patient_account_number",
    "pv_computed_los_days",
    "pv_has_insurance_balance",
).show(10, truncate=False)

print("Pipeline complete.")
