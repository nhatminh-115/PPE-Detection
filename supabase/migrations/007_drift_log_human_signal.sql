-- Migration 007: extend drift_log with explicit Scout/Human disagreement signals
-- Canonical retrain signal is human_disagreement_rate.

ALTER TABLE public.drift_log
  ADD COLUMN IF NOT EXISTS scout_total_auto        integer,
  ADD COLUMN IF NOT EXISTS scout_disagreements     integer,
  ADD COLUMN IF NOT EXISTS scout_disagreement_rate numeric(6,4),
  ADD COLUMN IF NOT EXISTS human_total_confirmed   integer,
  ADD COLUMN IF NOT EXISTS human_disagreements     integer,
  ADD COLUMN IF NOT EXISTS human_disagreement_rate numeric(6,4),
  ADD COLUMN IF NOT EXISTS rolling_7d_rate_scout   numeric(6,4);

COMMENT ON COLUMN public.drift_log.disagreement_rate IS
  'Canonical disagreement rate used for retrain trigger (human-confirmed).';
COMMENT ON COLUMN public.drift_log.scout_disagreement_rate IS
  'Scout-only disagreement rate: (has-fp+has-fn auto-labels) / total auto-labels.';
COMMENT ON COLUMN public.drift_log.human_disagreement_rate IS
  'Human-confirmed disagreement rate: (has-fp+has-fn confirmed labels) / total confirmed labels.';
