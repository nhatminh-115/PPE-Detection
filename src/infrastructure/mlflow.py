from src.config import MLFLOW_URI
import mlflow
import logging
import os

logger = logging.getLogger(__name__)


def find_latest_effnet_run_id(experiment_name: str, fallback: str) -> str:
    """
    Return the run_id of the most recent finished run that exported ONNX successfully.
    Falls back to `fallback` if MLflow is unavailable or no qualifying run exists.
    """
    if not MLFLOW_URI:
        return fallback
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.debug("MLflow experiment '%s' not found; using stable run", experiment_name)
            return fallback
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.export_onnx = 'success' AND status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            run_id = runs[0].info.run_id
            logger.info("MLflow: latest EffNet run with ONNX -> %s", run_id[:8])
            return run_id
    except Exception as exc:
        logger.warning("MLflow run discovery failed: %s", exc)
    return fallback


def pull_artifact_from_mlflow_run(run_id, artifact_path, cache_dir="model_cache"):
    """Fetches model artifacts idempotently from MLflow registry."""
    if not MLFLOW_URI:
        logger.warning("MLFLOW_TRACKING_URI is not set. Assuming local artifact execution.")
        return os.path.join(cache_dir, f"{run_id}_{os.path.basename(artifact_path)}")
        
    file_name = os.path.basename(artifact_path)
    cached_file_path = os.path.join(cache_dir, f"{run_id}_{file_name}")
    
    if os.path.exists(cached_file_path):
        logger.info(f"Cache HIT. Utilizing local artifact: {cached_file_path}")
        return cached_file_path
        
    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info(f"Cache MISS. Fetching artifact from remote Run ID: {run_id[:8]}...")
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=cache_dir)
    os.rename(downloaded_path, cached_file_path)
    
    logger.info(f"Artifact fetched and cached successfully.")
    return cached_file_path