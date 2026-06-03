import os
from datetime import datetime
from pathlib import Path
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, TimestampType


def _on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@pytest.fixture(scope="session")
def spark():
    if _on_databricks():
        s = SparkSession.builder.getOrCreate()
        s.conf.set("spark.sql.ansi.enabled", "false")
        yield s
        s.conf.set("spark.sql.ansi.enabled", "true")
    else:
        s = (SparkSession.builder.appName("rcm-unit-tests").master("local[2]")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false").getOrCreate())
        s.sparkContext.setLogLevel("ERROR")
        yield s
        s.stop()


@pytest.fixture
def test_tmp_dir(tmp_path: Path):
    if not _on_databricks():
        yield tmp_path
        return
    import uuid, shutil
    vol = Path("/Volumes/workspace/tirtho_db/tirtho_uploaded_files")
    run_dir = vol / f"_test_tmp_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yield run_dir
    shutil.rmtree(str(run_dir), ignore_errors=True)


def _spark_path(path) -> str:
    s = str(path)
    if _on_databricks() and not s.startswith("file:") and not s.startswith("/Volumes"):
        return f"file://{s}"
    return s


@pytest.fixture
def spark_path():
    return _spark_path


_CHARGES_COLS = [
    "rcm_client_id","rcm_npi","rcm_patient_account_number","rcm_unique_account_number",
    "org_code","patient_account_number","account_suffix","medical_record_number",
    "charge_id","empi_number","batch_number","seq_num","service_date",
    "charge_posting_date","amount","health_plan_type_code","charge_code",
    "charge_code_type","revenue_cost_center_code","detail_charge_quantity",
    "revenue_code","modifier1","modifier2","modifier3","charge_code_description",
    "number_of_units","patient_type_code","patient_type_code_description",
    "department","hcpcs_cpt_code","modifier4","modifier5","modifier6",
    "type_of_service","apc_code","status_indicator","apc_code_date",
    "discount_formula","composite_adjustment","hp1_non_covered_amount",
    "hp1_non_covered_qty","hp2_non_covered_amount","hp2_non_covered_qty",
    "hp3_non_covered_amount","hp3_non_covered_qty","exclude_charge_from_calc",
    "exclude_from_coins_ded_calc","national_drug_code","corrected_charge_flag",
    "corrected_charge_no","reposted_charge_no","place_of_service_code",
    "charge_dx_code1","charge_dx_code2","charge_dx_code3","charge_dx_code4",
    "charge_dx_code5","charge_dx_code6","charge_dx_code7","charge_dx_code8",
    "charge_dx_code9","charge_dx_code10","charge_dx_code_version",
    "charge_dx_code11","charge_dx_code12","cdm_proration_code",
    "cdm_prorated_revenue_code","late_charge","health_plan_number",
    "client_charge_code","financial_class_at_txn","service_provider_id",
    "billing_provider_id","file_date","host_system",
    "_ingestion_timestamp","_source_file",
]

_PATIENTVISITS_COLS = [
    "rcm_client_id","rcm_npi","rcm_patient_account_number","rcm_unique_account_number",
    "org_code","patient_account_number","account_suffix","medical_record_number",
    "empi_number","guarantor_number","patient_final_bill_date","patient_admission_date",
    "patient_discharge_date","patient_type_code","referring_provider_code",
    "admitting_provider_code","primary_care_provider_code","operating_provider_code",
    "attending_provider_code","financial_class_code","location_code","total_charges",
    "current_account_balance","insurance_balance","total_insurance_payments",
    "total_insurance_adjustments","patient_balance","total_patient_payments",
    "total_patient_adjustments","late_charges","unbilled_charges","account_status_code",
    "agency_code","collection_status","date_placed","next_follow_up_date",
    "first_mail_date","last_insurance_paid_date","last_patient_payment_amount",
    "last_patient_payment_date","outpatient_location","added_date","outlier_code",
    "number_of_benefit_days","claim_service_date","bill_date","bill_type_code",
    "bill_hold_flag","original_financial_class_code","bad_debt_collector_code",
    "accident_type_name","total_payments_from_collection_agency","type_of_admission",
    "bad_debt_transfer_date","admit_source_description","registrar_code",
    "sending_system_name","registration_date","medicare_remit_date","admission_code",
    "accident_date","patient_sub_type_group","discharge_code","discharge_room",
    "discharge_bed","length_of_stay","total_adjustments","unit_vip_level_code",
    "ur_charge_total","ar_charge_total","method_of_arrival",
    "discharge_nursing_station_code","discharge_nursing_station_name",
    "transfer_to_facility","pre_authorization_flag","coder_identifier","code_date",
    "total_time_worked_on_account","zero_balance_date","last_worked_date",
    "total_allowances","total_write_offs","credit_balance_date","service_code",
    "scheduled_service","type_of_service","total_payments","authorized_days",
    "admit_source_code","ub04_admit_type","calc_today_flag","archive_ind",
    "stop_contractual_allowance_extract_ind","ipf_per_diem_begin_day","eval_date",
    "referral_date","invoice_number","corrected_charge_flag","corrected_charge_no",
    "reposted_charge_no","charge_id","total_refunds","irf_value","case_mix_group_value",
    "patient_responsibility","vip_code_description","active_payer_indicator",
    "account_assign_date","provider_code","bad_debt_flag_yn","pre_admission_flag",
    "mothers_account_number","test_patient","current_health_plan_number",
    "appointment_made_date","expected_reimbursement_amount","file_date","host_system",
    "_ingestion_timestamp","_source_file",
]


def _build_schema(cols):
    return StructType([
        StructField(c, TimestampType() if c == "_ingestion_timestamp" else StringType(), True)
        for c in cols
    ])


BRONZE_CHARGES_SCHEMA       = _build_schema(_CHARGES_COLS)
BRONZE_PATIENTVISITS_SCHEMA = _build_schema(_PATIENTVISITS_COLS)


def _make_charge(**kwargs):
    defaults = {
        "rcm_client_id": "EXL001", "patient_account_number": "PA001",
        "charge_id": "CHG001", "service_date": "2024-01-15",
        "charge_posting_date": "2024-01-16", "amount": "250.00",
        "corrected_charge_flag": "N", "late_charge": "N",
        "patient_type_code": "O", "hcpcs_cpt_code": "99213",
        "charge_code": "C001", "charge_code_description": "Office Visit",
        "revenue_code": "0510", "number_of_units": "1", "host_system": "EPIC",
        "_ingestion_timestamp": datetime(2024, 1, 17, 8, 0, 0),
        "_source_file": "charges.csv",
    }
    row = {c: None for c in _CHARGES_COLS}
    row.update(defaults)
    row.update(kwargs)
    return tuple(row[c] for c in _CHARGES_COLS)


def _make_patientvisit(**kwargs):
    defaults = {
        "rcm_client_id": "EXL001", "patient_account_number": "PA001",
        "patient_admission_date": "2024-01-10",
        "patient_discharge_date": "2024-01-18",
        "patient_type_code": "I", "total_charges": "1800.00",
        "total_payments": "1500.00", "insurance_balance": "300.00",
        "patient_balance": "0.00", "length_of_stay": "8",
        "financial_class_code": "MC", "location_code": "LOC01",
        "host_system": "EPIC",
        "_ingestion_timestamp": datetime(2024, 1, 19, 8, 0, 0),
        "_source_file": "patientvisits.csv",
    }
    row = {c: None for c in _PATIENTVISITS_COLS}
    row.update(defaults)
    row.update(kwargs)
    return tuple(row[c] for c in _PATIENTVISITS_COLS)


@pytest.fixture
def make_charge():       return _make_charge
@pytest.fixture
def make_patientvisit(): return _make_patientvisit
@pytest.fixture
def bronze_charges_schema():       return BRONZE_CHARGES_SCHEMA
@pytest.fixture
def bronze_patientvisits_schema(): return BRONZE_PATIENTVISITS_SCHEMA


@pytest.fixture
def bronze_charges_df(spark):
    rows = [
        _make_charge(rcm_client_id="EXL001", charge_id="CHG001",
                     patient_account_number="PA001", amount="250.00"),
        _make_charge(rcm_client_id="EXL001", charge_id="CHG002",
                     patient_account_number="PA001", amount="1500.00"),
        _make_charge(rcm_client_id="EXL001", charge_id="CHG003",
                     patient_account_number="PA002", amount="75.50",
                     service_date="2024-02-05", charge_posting_date="2024-02-06"),
        _make_charge(rcm_client_id="EXL002", charge_id="CHG004",
                     patient_account_number="PA003", amount="12000.00",
                     service_date="2024-02-10", charge_posting_date="2024-02-11"),
        _make_charge(rcm_client_id="EXL001", charge_id="CHG005",
                     patient_account_number="PA002", amount="300.00",
                     service_date="2024-02-18", charge_posting_date="2024-02-19"),
    ]
    return spark.createDataFrame(rows, schema=BRONZE_CHARGES_SCHEMA)


@pytest.fixture
def bronze_patientvisits_df(spark):
    rows = [
        _make_patientvisit(rcm_client_id="EXL001", patient_account_number="PA001",
                           insurance_balance="300.00", length_of_stay="8"),
        _make_patientvisit(rcm_client_id="EXL001", patient_account_number="PA002",
                           patient_admission_date="2024-02-01",
                           patient_discharge_date="2024-02-03",
                           insurance_balance="1000.00", length_of_stay="2"),
        _make_patientvisit(rcm_client_id="EXL002", patient_account_number="PA003",
                           patient_admission_date="2024-02-10",
                           patient_discharge_date="2024-02-10",
                           insurance_balance="0.00", length_of_stay="0"),
    ]
    return spark.createDataFrame(rows, schema=BRONZE_PATIENTVISITS_SCHEMA)
