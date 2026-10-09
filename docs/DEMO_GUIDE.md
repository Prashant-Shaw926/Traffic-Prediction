# Demonstration guide

Use this order in a viva or supervisor demo. The application replays historical hours. It does not predict 2026, and it does not show a live feed.

Screenshots captured from the running app on 9 October 2026:

- [screenshots/prediction-2017-06-15.png](screenshots/prediction-2017-06-15.png)
- [screenshots/simulation-2017-06-15.png](screenshots/simulation-2017-06-15.png) — relationship 1 → 3 is lag 24 hours
- [screenshots/simulation-2017-06-25.png](screenshots/simulation-2017-06-25.png) — relationship 1 → 3 is lag 1 hour
- [screenshots/simulation-error-after-dataset-end.png](screenshots/simulation-error-after-dataset-end.png)

## If you need to capture screenshots yourself

## Before you start

From the repository root, in two terminals.

**Flask**

```powershell
$env:KERAS_BACKEND="torch"
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

Wait until the log says the Spatial GRU loaded and the server is on `http://127.0.0.1:5000`.

**React**

```powershell
npm run dev
```

Open `http://localhost:5173`. Use `localhost`, not `127.0.0.1`, if the second address refuses the connection.

## 1. Prediction

Stay on **Prediction**.

- Junction: junction 1.
- Start: `2017-06-15T10:00`.
- End: `2017-06-15T12:00`.

The window must be 1 to 2 hours. Click **Predict**.

Expect:

- `POST /api/predict` and HTTP 200.
- The inspector names the prediction model as Spatial GRU, 53 features, and lookback 168 hours.
- Charts show the forecast and the comparison with recorded counts.
- The map remains available.

Say clearly that this is a historical window inside the dataset, not today’s traffic.

## 2. Simulation at 15 June 2017, 10:00

Switch to **Historical Traffic Network Simulation**.

The timestamp field should already show `2017-06-15T10:00`. Click **Run Simulation**.

Expect HTTP 200 and:

- Four junction nodes. The layout is not a road map.
- Twelve directed relationships, one for every ordered pair.
- Separate figures for predicted traffic, historical traffic, and the influence index.
- Junction 1 is about 90.85 predicted vehicles, 90 historical vehicles, and an influence index of about 38.81. Those are different quantities.
- The selected relationship 1 → 2 has signed correlation about 0.931 and lag 1 hour.
- A legend states that line thickness is absolute correlation and the arrow is direction.

## 3. Simulation at 25 June 2017, 10:00

Change the timestamp to `2017-06-25T10:00` and run again.

Expect a new response, not the previous picture left in place. Junction 1 predicted traffic is about 50.78, historical traffic 52, influence index about 20.50.

Point at relationship **1 → 3**:

- At 15 June 10:00 the selected lag is **24 hours** (signed correlation about 0.635).
- At 25 June 10:00 the selected lag is **1 hour** (signed correlation about 0.592).

That change comes from recomputing lagged Pearson correlation on each window. It is recorded in `results/dynamic_simulation_stage7_report.txt`. It is not a stored road list.

Open a contribution line. It shows source traffic, the signed correlation, the lag, the absolute-correlation weight, and the contribution. The influence index is the sum of included contributions. It is not the predicted vehicle count.

## 4. An invalid timestamp

Show one failure. Either is enough:

- Set the hour to `2017-07-01T10:00` and run. Expect HTTP 400 and the text that the historical dataset ends at 2017-06-30 23:00.
- Or enable **Submit a raw timestamp**, type `not-a-datetime`, and run. Expect HTTP 400 and `Invalid datetime.`

The previous network must disappear. The error is the Flask message, not a substitute written only in the browser.

Also say why a date in 2026 is rejected: the file has no hours after 2017-06-30 23:00. The server is not claiming to forecast the current day.

## 5. What not to claim

- Do not say the arrows are surveyed roads.
- Do not say a positive correlation means one junction causes the other.
- Do not say Spatial GRU is better at every junction. On the June 2017 overlap test, junctions 3 and 4 have higher RMSE than the baseline GRU. See `docs/TECHNICAL_DOCUMENTATION.md`.
- Do not present the June demo screens as proof of a negative correlation. Those two hours selected positive correlations. Negative sign and missing-source exclusion are covered by `ml/simulation/tests/test_stage8.py`.

## If you need to capture screenshots yourself

1. Prediction view after a successful 15 June 10:00–12:00 forecast.
2. Simulation view after `2017-06-15T10:00`, including the relationship list.
3. Simulation view after `2017-06-25T10:00`, with 1 → 3 visible at lag 1.
4. The error text after `2017-07-01T10:00` or `not-a-datetime`.

Save them under `docs/screenshots/`. Do not draw a mock screen and label it as the application.
