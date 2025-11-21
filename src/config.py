import os

# --- Configuration ---
DATA_PATH = os.getenv("DATA_PATH", "data/Parkinsons_Speech-Features.csv")
SEED = int(os.getenv("SEED", 42))
