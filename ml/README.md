# Traffic prediction ML pipeline

Independent of the React UI and of Flask. Inspection, preprocessing, and **full LSTM training** are implemented. GRU / CNN-LSTM / ARIMA are not trained yet.

## Dataset

Place the Kaggle file at:

`data/raw/traffic.csv`

Source: [fedesoriano/traffic-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)

Do not rename columns. Do not overwrite `data/raw/`.

## Setup

From the repository root:

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r ml/requirements.txt
python -m ml.data.inspect_data
python -m ml.data.preprocess
python -m ml.training.train_lstm --smoke
python -m ml.training.train_lstm
```

If `data/raw/traffic.csv` is missing, that command exits with download instructions.

## Layout

| Path | Status |
| --- | --- |
| `ml/data/load_data.py` | implemented |
| `ml/data/inspect_data.py` | implemented |
| `ml/notebooks/01_dataset_inspection.ipynb` | implemented |
| `ml/data/preprocess.py` | implemented |
| `ml/data/clean_data.py` | implemented |
| `ml/features/` | implemented |
| `ml/training/sequence_generator.py` | implemented |
| `ml/models/lstm.py` | implemented |
| `ml/training/train_lstm.py` | full training + `--smoke` |
| `ml/evaluation/metrics.py` | implemented |
| `ml/inference/predictor.py` | implemented |
| GRU, CNN-LSTM, ARIMA | not trained yet |
