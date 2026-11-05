from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CROPS_DIR = ROOT / "data" / "crops"
MODEL_PATH = ROOT / "models" / "vehicle_classifier.joblib"

SPLITS = ("train", "valid", "test")
MIN_BOX_SIDE = 32