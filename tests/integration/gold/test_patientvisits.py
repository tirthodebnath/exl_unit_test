"""
tests/integration/gold/test_patientvisits.py
=============================================
PURPOSE:
    Visit-focused data quality tests on the real gold_rcm_summary_v2 table.

    These tests verify that the V2 join conditions were correctly applied.
    They answer: were only discharged patients joined? Did NPI and
    patient account number match correctly between charges and visits?

GOLD V2 JOIN CONDITIONS TESTED HERE:
    1. patient_discharge_date IS NOT NULL  → only discharged patients joined
    2. rcm_npi = rcm_npi                  → NPI must match
    3. patient_account_number = patient_account_number → account must match
    4. LEFT JOIN guarantee                → all charges kept even without visit

TESTS IN THIS FILE (visit-focused):
    TestGoldVisitJoinCorrectness
        1. test_discharge_date_not_null_where_visit_joined
        2. test_npi_matches_between_charge_and_visit
        3. test_patient_account_number_matches
        4. test_visit_columns_prefixed_pv
        5. test_left_join_keeps_all_charges
        6. test_rcm_client_id_not_null_in_gold
"""

import pytest
from pyspark.sql import functions as F


class TestGoldVisitJoinCorrectness:
    """
    Verifies that the V2 join conditions produced correct results on real data.

    The three join conditions (client_id + NPI + patient_account_number)
    plus the discharge date filter must all be enforced in the real output.
    """

    def test_discharge_date_not_null_where_visit_joined(self, gold_v2_real):
        """
        WHAT: Where visit columns are populated in Gold V2, the
              pv_patient_discharge_date must never be null.

        WHY: The join condition includes AND patient_discharge_date IS NOT NULL.
             This means only discharged patients are eligible to be matched.
             If any matched row has a null discharge date, the join condition
             was not enforced correctly — undischarged patients were joined.

        REAL DATA CHECK:
            Rows where pv_patient_account_number IS NOT NULL (visit matched)
            must ALL have pv_patient_discharge_date IS NOT NULL.
        """
        if "pv_patient_discharge_date" not in gold_v2_real.columns:
            pytest.skip("pv_patient_discharge_date column missing from Gold V2")

        # Find rows where a visit DID match (pv_patient_account_number is populated)
        # but discharge date is null — this would violate the join condition
        violated = gold_v2_real.filter(
            F.col("pv_patient_account_number").isNotNull()
            & F.col("pv_patient_discharge_date").isNull()
        ).count()

        assert violated == 0, (
            f"{violated} Gold V2 rows have a matched visit "
            f"(pv_patient_account_number is not null) but "
            f"pv_patient_discharge_date is null. "
            f"The discharge date IS NOT NULL join condition was not enforced."
        )

    def test_npi_matches_between_charge_and_visit(self, gold_v2_real):
        """
        WHAT: Where a visit was joined, the charge rcm_npi must equal
              the visit pv_rcm_npi.

        WHY: One of the V2 join conditions is c.NPI = pv.NPI. If any row
             in Gold has different NPI values between the charge and visit
             sides, the join condition was violated — the wrong visit was
             matched to this charge.

        REAL DATA CHECK:
            Rows where visit is joined: rcm_npi == pv_rcm_npi for all rows
        """
        if "pv_rcm_npi" not in gold_v2_real.columns:
            pytest.skip("pv_rcm_npi column missing from Gold V2")

        # Only check rows where a visit was actually joined
        joined_rows = gold_v2_real.filter(
            F.col("pv_patient_account_number").isNotNull()
        )

        npi_mismatch = joined_rows.filter(
            F.col("rcm_npi") != F.col("pv_rcm_npi")
        ).count()

        assert npi_mismatch == 0, (
            f"{npi_mismatch} Gold V2 rows have mismatched NPI between "
            f"charge (rcm_npi) and visit (pv_rcm_npi). "
            f"The NPI join condition was not enforced correctly."
        )

    def test_patient_account_number_matches(self, gold_v2_real):
        """
        WHAT: Where a visit was joined, the charge patient_account_number
              must equal the visit pv_patient_account_number.

        WHY: One of the V2 join conditions is
             c.PatientAccountNumber = pv.PatientAccountNumber.
             A mismatch means the wrong visit was matched to this charge,
             which would corrupt patient-level financial reporting.

        REAL DATA CHECK:
            Rows where visit is joined: patient_account_number == pv_patient_account_number
        """
        if "pv_patient_account_number" not in gold_v2_real.columns:
            pytest.skip("pv_patient_account_number column missing from Gold V2")

        joined_rows = gold_v2_real.filter(
            F.col("pv_patient_account_number").isNotNull()
        )

        account_mismatch = joined_rows.filter(
            F.col("patient_account_number") != F.col("pv_patient_account_number")
        ).count()

        assert account_mismatch == 0, (
            f"{account_mismatch} Gold V2 rows have mismatched patient_account_number "
            f"between charge and visit. "
            f"The PatientAccountNumber join condition was not enforced correctly."
        )

    def test_visit_columns_prefixed_pv(self, gold_v2_real):
        """
        WHAT: Key visit columns must be present with pv_ prefix in Gold V2.

        WHY: Both Silver tables share column names. The pv_ prefix prevents
             ambiguity and makes it clear which columns came from charges vs visits.
             If pv_ columns are missing, the join did not produce visit-side data.

        REAL DATA CHECK:
            pv_patient_account_number, pv_rcm_npi, pv_patient_discharge_date
            all exist in Gold V2 columns.
        """
        expected_pv_cols = [
            "pv_patient_account_number",
            "pv_rcm_npi",
            "pv_patient_discharge_date",
        ]
        missing = [c for c in expected_pv_cols if c not in gold_v2_real.columns]

        assert len(missing) == 0, (
            f"These pv_-prefixed visit columns are missing from Gold V2: {missing}. "
            f"Visit columns must be prefixed to avoid name ambiguity."
        )

    def test_left_join_keeps_all_charges(self, silver_and_gold_v2):
        """
        WHAT: Gold V2 row count must be >= Silver charges count.
              Every charge must appear at least once in Gold (LEFT JOIN guarantee).

        WHY: The V2 join has more conditions than V1 so fewer visits will match.
             But the LEFT JOIN means every charge still appears — with null
             pv_ columns if no visit matched all 3 conditions.
             If Gold has fewer distinct charge_ids than Silver, the LEFT JOIN
             accidentally became an INNER JOIN somewhere.

        REAL DATA CHECK:
            gold.select("charge_id").distinct().count() >= silver.count()
        """
        silver, gold = silver_and_gold_v2

        silver_count      = silver.count()
        gold_distinct_cnt = gold.select("charge_id").distinct().count()

        assert gold_distinct_cnt >= silver_count, (
            f"Gold V2 has {gold_distinct_cnt} distinct charge_ids but "
            f"Silver has {silver_count} charges. "
            f"LEFT JOIN must keep ALL charges — even those with no matching visit."
        )

    def test_rcm_client_id_not_null_in_gold(self, gold_v2_real):
        """
        WHAT: rcm_client_id must never be null in Gold V2.

        WHY: rcm_client_id comes from the charges side of the join.
             Since every charge has a non-null rcm_client_id (Silver filter
             removed nulls), Gold should never have a null rcm_client_id.
             A null here would mean a charge with no client — impossible
             after Silver filtering.

        REAL DATA CHECK:
            Zero rows where rcm_client_id is null in Gold V2
        """
        if "rcm_client_id" not in gold_v2_real.columns:
            pytest.skip("rcm_client_id column missing from Gold V2")

        null_client = gold_v2_real.filter(
            F.col("rcm_client_id").isNull()
        ).count()

        assert null_client == 0, (
            f"{null_client} rows have null rcm_client_id in Gold V2. "
            f"This should never happen — all charges must have a client ID "
            f"after Silver filtering."
        )
