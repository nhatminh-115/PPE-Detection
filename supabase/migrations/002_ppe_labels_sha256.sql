-- Phase 1: add crop_sha256 and fallback_used to ppe_labels
-- Run after 001_ppe_labels.sql.

ALTER TABLE ppe_labels
    ADD COLUMN IF NOT EXISTS crop_sha256   text,
    ADD COLUMN IF NOT EXISTS fallback_used boolean NOT NULL DEFAULT false;

CREATE UNIQUE INDEX IF NOT EXISTS ppe_labels_crop_sha256_idx ON ppe_labels (crop_sha256)
    WHERE crop_sha256 IS NOT NULL;
