-- =============================================================================
-- create_audit_table.sql
-- =============================================================================
-- PURPOSE:
--     Create the test_audit_log Delta table that stores the result of every
--     integration test run from test_after_load_data.py.
--
-- WHEN TO RUN:
--     Run this ONCE before running test_after_load_data.py for the first time.
--     After this, the audit table is populated automatically after every
--     test_after_load_data.py execution — no manual steps needed.
--
-- AUTO-UPDATE BEHAVIOUR:
--     The pytest hook in tests/integration/conftest.py fires after every
--     single test completes and appends one row to this table.
--     When you add a new test to tests/integration/, it automatically
--     appears in this table on the next run — no schema changes needed.
--
-- COLUMNS:
--     run_id           Unique ID per notebook execution (timestamp + UUID)
--                      Groups all tests from the same run together
--     run_timestamp    When the test run started (human readable)
--     layer            Which pipeline layer: bronze / silver / gold
--     table_name       Which table was being tested
--                      e.g. charges, patientvisits, silver_charges, gold_rcm_summary
--     test_class       The pytest class the test belongs to
--                      e.g. TestBronzeIngestion, TestGoldReconciliation
--     test_name        The individual test method name
--                      e.g. test_validate_file_format_compatibility
--     status           PASS / FAIL / SKIP
--     fail_message     null when status=PASS, full assertion error when FAIL
--                      This is the exact message from the failed assertion
--     notebook_link    Clickable URL to the test_after_load_data notebook
--     duration_seconds How long the individual test took to run
-- =============================================================================


-- Step 1: Create the audit log table
-- USING DELTA gives us time travel, ACID, and efficient append writes
-- IF NOT EXISTS means re-running this script will not error or overwrite data
CREATE TABLE IF NOT EXISTS workspace.tirtho_db.test_audit_log
(
    run_id           STRING    COMMENT 'Unique ID per notebook run — groups all tests from one execution',
    run_timestamp    STRING    COMMENT 'When this test run started e.g. 2024-01-15 10:30:00',
    layer            STRING    COMMENT 'Pipeline layer being tested: bronze / silver / gold',
    table_name       STRING    COMMENT 'Table being tested e.g. charges, silver_charges, gold_rcm_summary',
    test_class       STRING    COMMENT 'Pytest class name e.g. TestBronzeIngestion, TestGoldReconciliation',
    test_name        STRING    COMMENT 'Individual test method name e.g. test_validate_file_format_compatibility',
    status           STRING    COMMENT 'PASS / FAIL / SKIP',
    fail_message     STRING    COMMENT 'null when PASS, full assertion error message when FAIL',
    notebook_link    STRING    COMMENT 'Clickable URL to the test_after_load_data notebook that produced this row',
    duration_seconds FLOAT     COMMENT 'How long this individual test took in seconds'
)
USING DELTA
COMMENT 'Audit log of every integration test result from test_after_load_data.py.
         Automatically populated by pytest hooks — one row per test per run.
         New tests appear automatically on the next run without schema changes.';


-- Step 2: Verify table was created
DESCRIBE TABLE workspace.tirtho_db.test_audit_log;


-- Step 3: Useful queries after running test_after_load_data.py
-- ─────────────────────────────────────────────────────────────

-- See all results from the most recent run
-- SELECT * FROM workspace.tirtho_db.test_audit_log
-- WHERE run_id = (SELECT MAX(run_id) FROM workspace.tirtho_db.test_audit_log)
-- ORDER BY layer, table_name, test_name;

-- See all failed tests across all runs
-- SELECT run_timestamp, layer, table_name, test_name, fail_message
-- FROM workspace.tirtho_db.test_audit_log
-- WHERE status = 'FAIL'
-- ORDER BY run_timestamp DESC;

-- See pass rate per layer per run
-- SELECT run_id, run_timestamp, layer,
--        COUNT(*) as total_tests,
--        SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) as passed,
--        SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as failed
-- FROM workspace.tirtho_db.test_audit_log
-- GROUP BY run_id, run_timestamp, layer
-- ORDER BY run_timestamp DESC, layer;
