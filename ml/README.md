# Traffic prediction ML pipeline

Inspection, preprocessing, **full LSTM**, **full GRU**, **full CNN-LSTM**, Spatial GRU training, and model comparison are implemented. The Flask API in `backend/` serves Spatial GRU inference to the React app. ARIMA is not trained.

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
python -m ml.training.train_gru
python -m ml.training.train_cnn_lstm
python -m ml.evaluation.compare_models
pip install -r backend/requirements.txt
python -m flask --app backend.app run --host 127.0.0.1 --port 5000
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
| `ml/models/gru.py` | implemented |
| `ml/training/train_gru.py` | full training |
| `ml/models/cnn_lstm.py` | implemented |
| `ml/training/train_cnn_lstm.py` | full training |
| `ml/evaluation/metrics.py` | implemented |
| `ml/evaluation/compare_models.py` | LSTM / GRU / CNN-LSTM comparison (no retraining) |
| `ml/inference/predictor.py` | implemented |
| `backend/` | Flask API; `POST /api/predict` uses Spatial GRU |
| ARIMA | not trained yet |
