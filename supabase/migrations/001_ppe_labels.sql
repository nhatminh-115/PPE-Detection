-- Phase 1: Human-in-the-Loop Labeling
-- Run this migration in Supabase SQL editor or via supabase db push.

-- ── ppe_labels ───────────────────────────────────────────────────────────────
-- One row per (session_id, tracker_id) merge key.
-- Upserted on each label save; old state is written to ppe_label_audit first.

CREATE TABLE IF NOT EXISTS ppe_labels (
    id                  uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    merge_key           text        NOT NULL UNIQUE,
    session_id          text        NOT NULL,
    tracker_id          text        NOT NULL,

    -- Model metadata at time of labeling (for retrospective quality versioning)
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

    -- Normalized crop in Supabase Storage (224x224 JPEG)
    crop_url            text,
    crop_uploaded_at    timestamptz,

    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ppe_labels_session_idx   ON ppe_labels (session_id);
CREATE INDEX IF NOT EXISTS ppe_labels_aggregate_idx ON ppe_labels (aggregate);
CREATE INDEX IF NOT EXISTS ppe_labels_labeled_at_idx ON ppe_labels (labeled_at DESC);

-- Auto-update updated_at on each write
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS ppe_labels_updated_at ON ppe_labels;
CREATE TRIGGER ppe_labels_updated_at
    BEFORE UPDATE ON ppe_labels
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();


-- ── ppe_label_audit ──────────────────────────────────────────────────────────
-- Immutable append-only log of every label create/update/skip action.
-- Enables relabeling safety and multi-user conflict detection.

CREATE TABLE IF NOT EXISTS ppe_label_audit (
    id              uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    label_id        uuid        REFERENCES ppe_labels(id) ON DELETE SET NULL,
    merge_key       text        NOT NULL,
    action          text        NOT NULL,   -- 'create' | 'update' | 'skip' | 'unskip'
    previous_state  jsonb,
    new_state       jsonb       NOT NULL,
    labeled_by      text        NOT NULL DEFAULT 'anonymous',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ppe_label_audit_merge_key_idx ON ppe_label_audit (merge_key);
CREATE INDEX IF NOT EXISTS ppe_label_audit_created_at_idx ON ppe_label_audit (created_at DESC);

-- Row-level security: allow service role full access, anon read-only for labels
ALTER TABLE ppe_labels      ENABLE ROW LEVEL SECURITY;
ALTER TABLE ppe_label_audit ENABLE ROW LEVEL SECURITY;

-- Service role bypass (used by backend)
CREATE POLICY "service_full_labels"      ON ppe_labels      FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_full_audit"       ON ppe_label_audit FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Anon read-only for dashboard display
CREATE POLICY "anon_read_labels"         ON ppe_labels      FOR SELECT TO anon      USING (true);
CREATE POLICY "anon_read_audit"          ON ppe_label_audit FOR SELECT TO anon      USING (true);
