-- PPE Detection System: Complete Database Schema
-- Run this consolidated file for quick setup, or use numbered migrations (001-007) for version control.
-- Generated from migrations 001-007.

-- ───────────────────────────────────────────────────────────────────────────────
-- CORE EVENT & LABEL TABLES
-- ───────────────────────────────────────────────────────────────────────────────

-- ppe_labels: Current annotation state, one row per (session_id, tracker_id) merge key.
-- Upserted on each save; includes both human and auto-labeled verdicts.
CREATE TABLE IF NOT EXISTS public.ppe_labels (
    id                  uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    merge_key           text        NOT NULL UNIQUE,
    session_id          text        NOT NULL,
    tracker_id          text        NOT NULL,

    -- Model metadata at time of labeling
    model_name          text,
    model_version       text,
    run_id              text,
    threshold_snapshot  jsonb       NOT NULL DEFAULT '{}',

    -- Violation classification
    predicted_missing   text[]      NOT NULL DEFAULT '{}',
    human_missing       text[],
    per_item            jsonb       NOT NULL DEFAULT '{}',
    aggregate           text,           -- 'all-tp' | 'has-fp' | 'has-fn' | 'skip'
    skipped             boolean     NOT NULL DEFAULT false,

    -- Source event references
    model               text,
    row_ids             text[]      NOT NULL DEFAULT '{}',
    first_created_at    timestamptz,

    -- Annotation provenance
    labeled_by          text        NOT NULL DEFAULT 'anonymous',
    labeled_at          timestamptz NOT NULL DEFAULT now(),

    -- Normalized crop (224x224 JPEG) in Supabase Storage
    crop_url            text,
    crop_sha256         text,   -- Deduplicate by SHA-256 hash
    crop_uploaded_at    timestamptz,

    -- Auto-labeling signals
    is_auto_labeled     boolean     NOT NULL DEFAULT false,
    auto_label_model    text,       -- 'llama-4-scout' | null
    fallback_used       boolean     NOT NULL DEFAULT false,  -- Real-ESRGAN fallback

    -- Retention policy
    expire_at           timestamptz,  -- 2 days (auto) | 3 days (human)

    -- Audit trail
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ppe_labels_session_idx   ON public.ppe_labels (session_id);
CREATE INDEX IF NOT EXISTS ppe_labels_aggregate_idx ON public.ppe_labels (aggregate);
CREATE INDEX IF NOT EXISTS ppe_labels_labeled_at_idx ON public.ppe_labels (labeled_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS ppe_labels_crop_sha256_idx ON public.ppe_labels (crop_sha256)
    WHERE crop_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS ppe_labels_auto_idx ON public.ppe_labels (is_auto_labeled);
CREATE INDEX IF NOT EXISTS ppe_labels_expire_idx ON public.ppe_labels (expire_at)
    WHERE expire_at IS NOT NULL;

-- Auto-update updated_at on writes
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS ppe_labels_updated_at ON public.ppe_labels;
CREATE TRIGGER ppe_labels_updated_at
    BEFORE UPDATE ON public.ppe_labels
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();


-- ppe_label_audit: Immutable change log of all label mutations (create, update, skip).
CREATE TABLE IF NOT EXISTS public.ppe_label_audit (
    id              uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    label_id        uuid        REFERENCES public.ppe_labels(id) ON DELETE SET NULL,
    merge_key       text        NOT NULL,
    action          text        NOT NULL,   -- 'create' | 'update' | 'skip' | 'unskip'
    previous_state  jsonb,
    new_state       jsonb       NOT NULL,
    labeled_by      text        NOT NULL DEFAULT 'anonymous',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ppe_label_audit_merge_key_idx ON public.ppe_label_audit (merge_key);
CREATE INDEX IF NOT EXISTS ppe_label_audit_created_at_idx ON public.ppe_label_audit (created_at DESC);


-- ppe_violations: Raw detection events logged by inference pipeline.
-- Fire-and-forget log sink; used for audit trail, not primary decision table.
CREATE TABLE IF NOT EXISTS public.ppe_violations (
    id              uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      text        NOT NULL,
    tracker_id      text        NOT NULL,
    timestamp       timestamptz NOT NULL DEFAULT now(),
    violation_data  jsonb       NOT NULL,
    INDEX (session_id, tracker_id, timestamp DESC)
);


-- ───────────────────────────────────────────────────────────────────────────────
-- REPORTING & OBSERVABILITY TABLES
-- ───────────────────────────────────────────────────────────────────────────────

-- daily_reports: End-of-day compliance snapshot; upserted by report_date.
CREATE TABLE IF NOT EXISTS public.daily_reports (
    id              uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    report_date     date        NOT NULL UNIQUE,
    
    -- LangGraph pipeline output
    narrative       text,
    executive_summary text,
    
    -- Regulation-cited details
    violations_by_type jsonb,  -- { "missing_hardhat": { "count": 5, "citations": [...] }, ... }
    corrective_actions text[],
    
    -- Quality metrics snapshot
    quality         jsonb,     -- { "precision": 0.92, "recall": 0.88, "tp": 142, "fp": 12, "fn": 19 }
    systemic_flags  text[]     NOT NULL DEFAULT '{}',  -- Risk patterns detected
    
    -- Production metadata
    engine          text       NOT NULL DEFAULT 'langgraph',  -- 'langgraph' | 'legacy'
    
    -- Telegram notification
    telegram_sent   boolean    DEFAULT false,
    telegram_file_id text,
    
    -- Artifact tracking
    s3_path         text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS daily_reports_date_idx ON public.daily_reports (report_date DESC);


-- drift_log: Model drift monitoring; one row per day.
-- Canonical retrain signal: human_disagreement_rate > 0.25 (rolling 7-day avg).
CREATE TABLE IF NOT EXISTS public.drift_log (
    id                      uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    date                    date        UNIQUE NOT NULL,

    -- Scout disagreement (early warning signal)
    scout_total_auto        integer,
    scout_disagreements     integer,
    scout_disagreement_rate numeric(6,4),

    -- Human disagreement (canonical retrain trigger)
    human_total_confirmed   integer,
    human_disagreements     integer,
    human_disagreement_rate numeric(6,4),

    -- Rolling averages
    rolling_7d_rate_scout   numeric(6,4),
    rolling_7d_rate         numeric(6,4),     -- human disagreement 7-day rolling avg

    -- Action taken
    retrain_triggered       boolean     DEFAULT false,
    computed_at             timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS drift_log_date_idx ON public.drift_log (date DESC);

COMMENT ON TABLE public.drift_log IS
  'Daily model disagreement tracking. human_disagreement_rate rolling 7d avg > 0.25 triggers EfficientNet retrain.';
COMMENT ON COLUMN public.drift_log.scout_disagreement_rate IS
  '(has-fp + has-fn auto-labels) / total Scout auto-labels on this date. Early warning signal.';
COMMENT ON COLUMN public.drift_log.human_disagreement_rate IS
  '(has-fp + has-fn confirmed human labels) / total confirmed labels. Canonical retrain signal.';


-- ───────────────────────────────────────────────────────────────────────────────
-- ROW-LEVEL SECURITY (RLS)
-- ───────────────────────────────────────────────────────────────────────────────

ALTER TABLE public.ppe_labels      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ppe_label_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ppe_violations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_reports   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drift_log       ENABLE ROW LEVEL SECURITY;

-- Backend (service_role key) has full access
CREATE POLICY "service_full_labels"      ON public.ppe_labels      FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_full_audit"       ON public.ppe_label_audit FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_full_violations"  ON public.ppe_violations  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_full_reports"     ON public.daily_reports   FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_full_drift"       ON public.drift_log       FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Anonymous users (dashboard) have read-only access
CREATE POLICY "anon_read_labels"         ON public.ppe_labels      FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_audit"          ON public.ppe_label_audit FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_violations"     ON public.ppe_violations  FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_reports"        ON public.daily_reports   FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_drift"          ON public.drift_log       FOR SELECT TO anon USING (true);
