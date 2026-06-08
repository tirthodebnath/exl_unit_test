-- =============================================================================
-- create_metadata_table.sql
-- =============================================================================
-- PURPOSE:
--     Create and populate the test_table_metadata Delta table.
--     This table drives parametrized testing in test_after_load_data.py.
--     Instead of hardcoding table names and config in notebooks, we read
--     them from here — making the test suite fully dynamic.
--
-- WHEN TO RUN:
--     Run this ONCE before running any notebooks.
--     After this, test_after_load_data.py will read from this table.
--
-- HOW TO RUN:
--     Open a new Databricks notebook, paste this SQL, and run each cell.
--     Or use the Databricks SQL editor.
--
-- COLUMN DESCRIPTIONS:
--     table_name         Logical name used in test output (e.g. "charges")
--     filename           Source CSV filename (e.g. "charges.csv")
--     key_col            Primary key column for null/dedup checks
--     source_path        Full Volume path where the source CSV lives
--     bronze_table       Fully qualified Bronze Delta table name
--     silver_table       Fully qualified Silver Delta table name
--     expected_col_count Number of data columns (excluding _ingestion_timestamp
--                        and _source_file audit columns)
--     active             Set to true to include in test runs, false to skip
-- =============================================================================


-- Step 1: Create the metadata table
-- IF NOT EXISTS means running this script twice will not error
CREATE TABLE IF NOT EXISTS workspace.tirtho_db.test_table_metadata
(
    table_name          STRING   COMMENT 'Logical name of the table e.g. charges',
    filename            STRING   COMMENT 'Source CSV filename e.g. charges.csv',
    key_col             STRING   COMMENT 'Primary key column name for null/dedup checks',
    source_path         STRING   COMMENT 'Full Volume path to the source CSV file',
    bronze_table        STRING   COMMENT 'Fully qualified Bronze Delta table name',
    silver_table        STRING   COMMENT 'Fully qualified Silver Delta table name',
    expected_col_count  INT      COMMENT 'Expected number of data columns excluding audit cols',
    active              BOOLEAN  COMMENT 'Set true to include in test runs, false to skip'
)
USING DELTA
COMMENT 'Metadata table that drives parametrized RCM pipeline tests in test_after_load_data.py';


-- Step 2: Insert the two RCM table configurations
-- If you add a new source table to the pipeline, add a new row here.
-- test_after_load_data.py will automatically pick it up on the next run.
INSERT INTO workspace.tirtho_db.test_table_metadata
VALUES
(
    'charges',                  -- table_name
    'charges.csv',              -- filename
    'charge_id',                -- key_col (primary key for charges)
    '/Volumes/workspace/tirtho_db/tirtho_uploaded_files/charges.csv',
    'workspace.tirtho_db.bronze_charges',
    'workspace.tirtho_db.silver_charges',
    73,                         -- number of data columns in charges schema
    true                        -- active = include in test runs
),
(
    'patientvisits',            -- table_name
    'patientvisits.csv',        -- filename
    'patient_account_number',   -- key_col (primary key for patientvisits)
    '/Volumes/workspace/tirtho_db/tirtho_uploaded_files/patientvisits.csv',
    'workspace.tirtho_db.bronze_patientvisits',
    'workspace.tirtho_db.silver_patientvisits',
    111,                        -- number of data columns in patientvisits schema
    true                        -- active = include in test runs
);


-- Step 3: Verify the rows were inserted correctly
SELECT
    table_name,
    filename,
    key_col,
    source_path,
    bronze_table,
    silver_table,
    expected_col_count,
    active
FROM workspace.tirtho_db.test_table_metadata
ORDER BY table_name;
