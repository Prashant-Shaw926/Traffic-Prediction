from pathlib import Path

# ml/data/load_data.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = REPO_ROOT / "data" / "raw" / "traffic.csv"

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset"
)

DOWNLOAD_INSTRUCTIONS = f"""Kaggle traffic.csv was not found at:
  {RAW_CSV_PATH}

Download the real dataset (do not invent a CSV):

  1. Open {KAGGLE_DATASET_URL}
  2. Click Download
  3. The archive contains traffic.csv (the only data file needed)
  4. Place it at:
       data/raw/traffic.csv

If you downloaded a zip, extract it so that path is exact.
Do not rename columns. Do not overwrite data/raw/ after placing the file.
"""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when data/raw/traffic.csv is missing."""


def load_raw_traffic():
    """Load the unmodified Kaggle CSV. Never synthesizes rows."""
    import pandas as pd

    if not RAW_CSV_PATH.is_file():
        raise DatasetNotFoundError(DOWNLOAD_INSTRUCTIONS)

    return pd.read_csv(RAW_CSV_PATH)
