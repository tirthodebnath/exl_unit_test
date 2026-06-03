"""
Bronze unit tests — parametrized across both tables using table metadata.

TABLE_METADATA drives both test cases so adding a new table only requires
one new entry here, no new test functions.
"""
import pytest
from src.bronze.ingest import ingest_charges, ingest_patientvisits
from src.common.schemas import CHARGES_BRONZE_SCHEMA, PATIENTVISITS_BRONZE_SCHEMA

# ---------------------------------------------------------------------------
# Table metadata — key col differs per table, everything else is the same contract
# ---------------------------------------------------------------------------
TABLE_METADATA = [
    pytest.param(
        {
            "name":       "charges",
            "ingest_fn":  ingest_charges,
            "schema":     CHARGES_BRONZE_SCHEMA,
            "filename":   "charges.csv",
            "key_col":    "charge_id",          # null-key test uses this column
        },
        id="charges",
    ),
    pytest.param(
        {
            "name":       "patientvisits",
            "ingest_fn":  ingest_patientvisits,
            "schema":     PATIENTVISITS_BRONZE_SCHEMA,
            "filename":   "patientvisits.csv",
            "key_col":    "patient_account_number",
        },
        id="patientvisits",
    ),
]


def _write_csv(path, schema, rows_override=None):
    """Write a minimal CSV matching the given schema."""
    data_cols = [f.name for f in schema.fields
                 if f.name not in ("_ingestion_timestamp", "_source_file")]
    header = ",".join(data_cols)
    if rows_override:
        body = "\n".join(rows_override)
    else:
        body = ",".join(["TEST"] * len(data_cols))
    path.write_text(f"{header}\n{body}\n")


@pytest.mark.parametrize("meta", TABLE_METADATA)
class TestBronzeNullKeys:
    """
    Parametrized: runs once for charges (key=charge_id)
                  and once for patientvisits (key=patient_account_number).
    """

    def test_null_key_rows_land_in_bronze(self, spark, test_tmp_dir,
                                          spark_path, meta):
        """
        Bronze must preserve ALL rows including those with a null key column.
        Filtering is Silver's job — Bronze is a faithful copy of the source.
        """
        schema   = meta["schema"]
        key_col  = meta["key_col"]
        data_cols = [f.name for f in schema.fields
                     if f.name not in ("_ingestion_timestamp", "_source_file")]

        # Row with null key (blank CSV field → null with explicit schema)
        null_row  = ",".join(
            "" if col == key_col else "TEST" for col in data_cols
        )
        # Row with valid key
        valid_row = ",".join(["TEST"] * len(data_cols))

        csv = test_tmp_dir / meta["filename"]
        csv.write_text(",".join(data_cols) + "\n" + null_row + "\n" + valid_row + "\n")

        df = meta["ingest_fn"](spark, spark_path(csv))

        # Both rows must be present — Bronze never drops
        assert df.count() == 2

        # The null key row must have a null value in the key column
        null_count = df.filter(df[key_col].isNull()).count()
        assert null_count == 1, (
            f"Expected 1 null {key_col} row in Bronze, got {null_count}"
        )

    def test_no_exact_duplicate_rows(self, spark, test_tmp_dir,
                                     spark_path, meta):
        """
        Ingesting a file with unique rows must not produce duplicate rows.
        Bronze should pass through what it reads without multiplying rows.
        """
        schema    = meta["schema"]
        data_cols = [f.name for f in schema.fields
                     if f.name not in ("_ingestion_timestamp", "_source_file")]

        # Three distinct rows
        rows = [
            ",".join([f"VAL{i}"] * len(data_cols))
            for i in range(1, 4)
        ]
        csv = test_tmp_dir / meta["filename"]
        csv.write_text(",".join(data_cols) + "\n" + "\n".join(rows) + "\n")

        df = meta["ingest_fn"](spark, spark_path(csv))

        total    = df.count()
        # Exclude _ingestion_timestamp from dedup check — same batch = same ts
        dedup_cols = [c for c in df.columns if c != "_ingestion_timestamp"]
        distinct = df.dropDuplicates(dedup_cols).count()

        assert total == distinct == 3, (
            f"Expected 3 unique rows, got total={total} distinct={distinct}"
        )
