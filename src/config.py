from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
TWO_TOWER_DIR = MODELS_DIR / "two_tower"
RANKER_DIR = MODELS_DIR / "ranker"
E5_DIR = MODELS_DIR / "e5"

for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TWO_TOWER_DIR, RANKER_DIR, E5_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# synthetic sizes
N_USERS = 50_000
N_PRODUCTS = 2_000
N_INTERACTIONS = 500_000

RANDOM_SEED = 42

# two-tower config
EMBEDDING_DIM = 64
INFO_NCE_TEMPERATURE = 0.07
BATCH_SIZE = 1024
N_EPOCHS = 3
LR_TWO_TOWER = 1e-3

