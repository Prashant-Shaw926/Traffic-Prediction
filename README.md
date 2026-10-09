# Traffic Prediction Application: A Data-Driven Approach Using Deep Learning

Historical traffic prediction for four junctions, plus a historical traffic-network simulation. The running application uses a trained Spatial GRU. LSTM, GRU, and CNN-LSTM remain as completed baseline experiments. This is not a live traffic feed. The series ends at **2017-06-30 23:00**.

## What the app does

- **Prediction.** Choose a junction and a 1–2 hour historical window. The UI calls `POST /api/predict` and shows the Spatial GRU forecast against recorded counts.
- **Historical Traffic Network Simulation.** Choose one historical hour. Python recalculates lagged Pearson relationships and a formula-based influence index, predicts each junction with the same Spatial GRU, and the UI draws that response.

Relationships are lagged statistical associations. They are not roads and they are not causes. The network drawing is a display layout, not a map.

## Stack

- React, TypeScript, Vite, Tailwind CSS
- Flask REST API
- Keras 3 with the PyTorch backend
- Trained artifacts under `models/`; no retraining at runtime

## Setup (Windows)

From the repository root, with Python available for the virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm install
```

Copy `.env.example` to `.env` if needed. The frontend expects:

```text
VITE_API_BASE_URL=/api
```

Vite proxies `/api` to Flask at `http://127.0.0.1:5000`. Start both processes.

**Terminal 1 — Flask**

```powershell
$env:KERAS_BACKEND="torch"
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

**Terminal 2 — React**

```powershell
npm run dev
```

Open `http://localhost:5173`. If only Vite is running, the junction list stays empty.

## Demo times

- Prediction: junction 1, 2017-06-15 10:00 to 2017-06-15 12:00.
- Simulation: `2017-06-15T10:00`, then `2017-06-25T10:00`.
- A timestamp after 2017-06-30 23:00, including a 2026 date, is rejected. The dataset does not contain it.

## Further reading

- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
- [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)
- [docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md)
- [backend/README.md](backend/README.md) for the API
- [ml/README.md](ml/README.md) for the offline training pipeline

Dataset file: `data/raw/traffic.csv`, from the [Kaggle traffic prediction dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset). Do not overwrite `data/raw/`.
