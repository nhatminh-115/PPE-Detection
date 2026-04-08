-- Phase 3: mark auto-labeled rows produced by the second opinion agent.
-- is_auto_labeled=true  → labeled by a model (e.g. Llama 4 Scout), not a human.
-- auto_label_model      → which model produced the label (e.g. 'llama-4-scout').
-- Only is_auto_labeled=false rows are used for precision/recall in daily quality reports.

ALTER TABLE ppe_labels
    ADD COLUMN IF NOT EXISTS is_auto_labeled  boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS auto_label_model text;

CREATE INDEX IF NOT EXISTS ppe_labels_auto_idx ON ppe_labels (is_auto_labeled);
