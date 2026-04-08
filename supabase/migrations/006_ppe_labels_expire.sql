-- Workstream B: retention policy for crop storage.
-- expire_at is set at label creation time:
--   is_auto_labeled=true  (Scout)  → now + 2 days
--   is_auto_labeled=false (Human)  → now + 3 days
-- Cleanup routine nulls out crop_url/crop_sha256 for expired rows
-- and removes the corresponding objects from Supabase Storage.

ALTER TABLE ppe_labels
    ADD COLUMN IF NOT EXISTS expire_at timestamptz;

CREATE INDEX IF NOT EXISTS ppe_labels_expire_idx
    ON ppe_labels (expire_at)
    WHERE expire_at IS NOT NULL;
