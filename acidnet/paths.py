import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("ACIDNET_DATASET_ROOT", PROJECT_ROOT / "datasets"))
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
WEIGHT_ROOT = PROJECT_ROOT / "experiments" / "weights"
