-- Phase 2: add quality metrics and systemic flags to daily_reports
-- Run after 002_ppe_labels_sha256.sql.
--
-- quality: jsonb snapshot from get_daily_quality (precision, recall, TP/FP/FN counts).
-- systemic_flags: array of human-readable risk notes from pattern_agent.
-- engine: which pipeline produced the report ('legacy' | 'langgraph').

ALTER TABLE daily_reports
    ADD COLUMN IF NOT EXISTS quality        jsonb,
    ADD COLUMN IF NOT EXISTS systemic_flags text[]  NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS engine         text    NOT NULL DEFAULT 'legacy';
