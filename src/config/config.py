from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_INTERIM = ROOT_DIR / "data" / "interim"
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_PREDICTIONS = ROOT_DIR / "data" / "predictions"
DATA_FEATURES = ROOT_DIR / "data" / "features"
DATA_EXPORTS = ROOT_DIR / "data" / "exports"

MODELS = ROOT_DIR / "models" 
MODELS_BASELINE = ROOT_DIR / "models" / "baseline"
MODELS_TUNED = ROOT_DIR / "models" / "tuned"
MODELS_METRICS = ROOT_DIR / "models" / "metrics"
MODELS_PARAM = ROOT_DIR / "models" / "param"
MODELS_ANOMALYD = ROOT_DIR / "models" / "anomalydetection"
MODELS_PRODUCTION = ROOT_DIR / "models" / "production"

RANDOM_STATE = 42
TEST_SIZE = 0.2

HI_FEATURES = ["s3_std", "s3_var", "s4_shape_factor"]