from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELING_DATA_DIR = DATA_DIR / "modeling"
MODEL_RESULTS_DIR = DATA_DIR / "model_results"
VISUALS_DIR = PROJECT_ROOT / "visuals"
CHARTS_DIR = VISUALS_DIR / "charts"
MEDIUM_VISUALS_DIR = VISUALS_DIR / "medium"
