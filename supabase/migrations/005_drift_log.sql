-- Migration 005: drift_log table for Phase 4 disagreement-rate monitoring
-- Each row records one day's Scout disagreement stats.
-- rolling_7d_rate > DRIFT_THRESHOLD (0.25) triggers EfficientNet retrain.

CREATE TABLE IF NOT EXISTS public.drift_log (
  id                  uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  date                date UNIQUE NOT NULL,
  total_auto          integer NOT NULL DEFAULT 0,
  disagreements       integer NOT NULL DEFAULT 0,
  disagreement_rate   numeric(6,4),
  rolling_7d_rate     numeric(6,4),
  retrain_triggered   boolean DEFAULT false,
  computed_at         timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS drift_log_date_idx ON public.drift_log (date DESC);

COMMENT ON TABLE public.drift_log IS
  'Daily Scout disagreement rate. rolling_7d_rate > 0.25 triggers EfficientNet retrain.';
COMMENT ON COLUMN public.drift_log.disagreement_rate IS
  '(has-fp + has-fn auto-labels) / total auto-labels on this date.';
COMMENT ON COLUMN public.drift_log.rolling_7d_rate IS
  'Average disagreement_rate over the last 7 days including today.';
