# Viva questions

Answers describe this repository. Items under future work are not built.

## 1. Why was this problem selected?

Hourly vehicle counts at a junction are a sequence: the recent past, the same hour yesterday, and the same hour last week are informative. The project asks whether a deep sequence model can estimate the next count, and whether lagged counts at the other junctions add information. The delivered application also shows those lagged statistical associations at a chosen historical hour.

## 2. What is the dataset and its limitation?

`data/raw/traffic.csv` has 48,120 rows and the columns `DateTime`, `Junction`, `Vehicles`, and `ID`. There are four junctions. Counts are hourly. Junctions 1–3 run from 2015-11-01 00:00 through 2017-06-30 23:00. Junction 4 starts on 2017-01-01. The target is `Vehicles`. `ID` is not a feature. There is no weather, holiday flag, speed, occupancy, GPS, or road map in the file. The series stops at 2017-06-30 23:00.

## 3. Why use a time-series model?

The target is an ordered hourly count. A model that sees the previous 168 hours can use daily and weekly structure. A shuffled row model would break that order. The split is chronological for the same reason.

## 4. What are LSTM, GRU, and CNN-LSTM?

All three are trained here as baseline models on 13 features and 168-hour windows.

- LSTM: LSTM 128 with sequence output, dropout 0.2, LSTM 64, dense 32 ReLU, dense 1 linear.
- GRU: the same stack with GRU cells instead of LSTM cells.
- CNN-LSTM: Conv1D with 32 filters and kernel size 3, max pooling, LSTM 128, dense 32 ReLU, dense 1 linear.

The GRU cell is a gated recurrent unit. It keeps a hidden state over the lookback window. The CNN-LSTM first applies a short convolution along that window, then an LSTM.

## 5. Why was GRU selected among the baseline models?

On the original baseline test split (`results/model_comparison.csv`), GRU has RMSE 5.101930, MAE 3.207288, and R² 0.966585. Those are the best RMSE, MAE, and R² of the three. CNN-LSTM has a lower MAPE (13.359807% versus GRU 13.479096%). GRU was selected on RMSE, MAE, and R², not because it won every metric.

## 6. Why did the Spatial GRU use 53 features?

The baseline 13 features are calendar fields, cyclic hour, junction id, and the junction’s own lags at 1, 2, 3, 24, and 168 hours. The spatial set adds lagged network totals and averages, lagged traffic at each of the four junctions, and lagged other-junction traffic. The feature list in `models/artifacts/spatial/feature_config.json` has 53 names. Contemporaneous `Vehicles` at time T is not one of them. The ablation report compares 13, 23, 43, and 53 features on the same June 2017 test window.

## 7. What is the 168-hour lookback?

Each prediction reads 168 consecutive hourly rows, one week, ending at the hour being predicted. Lag 168 is the same idea for a single feature: the count one week earlier. The first 168 hours of a series are dropped because that lag does not exist yet. For the spatial model, the first usable hour after Junction 4 joins is 2017-01-08 00:00.

## 8. How did you prevent target leakage?

`Vehicles` at the prediction hour is the target, not an input. Features use lags through T−1. The scaler is fit on the training rows only. The split is chronological and is not shuffled. The spatial service raises if `Vehicles` appears in the feature list or if `Vehicles(T)` is copied into the tensor. Historical traffic shown in the UI is the recorded count for comparison after prediction. The React client does not feed that count back into the model.

## 9. What do RMSE, MAE, MAPE, and R² measure?

- RMSE is the square root of the mean squared error, in vehicles. Large misses count more.
- MAE is the mean absolute error, in vehicles.
- MAPE is the mean absolute percentage error. It divides by the actual count.
- R² compares the residual sum of squares with the variance of the actual counts. Higher is a closer fit on that split. It is not a probability.

Lower RMSE, MAE, and MAPE are better individually. Higher R² is better individually. This project does not collapse them into one score.

## 10. Why can MAPE be misleading for low traffic counts?

MAPE divides by the actual count. A miss of two vehicles is a large percentage when the hour has only a few vehicles, and a small percentage in a busy hour. In the same-period report, junction 4 MAPE is about 30% while junction 1 MAPE is about 5–7%, even though junction 4’s RMSE is smaller than junction 1’s. CNN-LSTM can have a better MAPE than GRU while GRU has a better RMSE. That is why model choice was not based on MAPE alone.

## 11. What is lagged Pearson correlation?

For a lag of k hours, the simulation pairs the source junction’s count at time t−k with the target junction’s count at time t, inside the 168-hour window ending at the requested hour. The Pearson correlation of those pairs is `C(source → target, k)`. Pairs are matched by timestamp. A missing hour is skipped, not filled. The lag with the largest absolute correlation is kept. Candidate lags are 1, 2, 3, 24, and 168. At least 24 pairs are required.

## 12. Why preserve the correlation sign but use its absolute value as the influence weight?

The sign says whether the two series moved together or in opposite directions. The influence formula uses a non-negative weight so the contribution is `absolute_correlation × source count`, not a signed cancellation. The signed value is still stored and shown. If two lags have the same absolute correlation, the earlier lag in the configured list is kept. Stage 8 tests this with synthetic series. The June demonstration hours happened to select positive correlations.

## 13. What is the difference between predicted traffic and the influence index?

Predicted traffic is the Spatial GRU’s vehicle-count estimate. Historical traffic is the recorded count at that hour, when the file has one. The influence index is the sum of included contributions. Each contribution is an absolute correlation times another junction’s count at T−lag. It is not a vehicle count and it is not the forecast. At 2017-06-15 10:00, junction 1 is about 90.85 predicted vehicles, 90 recorded vehicles, and an influence index of about 38.81.

## 14. Why are relationships not proof of physical road connectivity?

The dataset has no GPS and no road graph. A high correlation means the two hourly series moved together at some lag inside that window. They might share a time-of-day pattern without a direct road between them. The arrow in the UI is the statistical direction used in the formula, source to target. The node positions are a display layout.

## 15. Why does the application reject current dates?

The last hour in the file is 2017-06-30 23:00. A timestamp after that, including a date in 2026, has no history to build the lookback or the simulation window. Flask returns HTTP 400 with the dataset-end message. That is intentional. The app is not a live forecast service.

## 16. What is the difference between offline training and runtime inference?

Training scripts under `ml/training/` fit models and scalers and write files under `models/`. That step is finished. When Flask starts, it loads the saved Spatial GRU and scalers. A user request runs inference only. The simulation also recomputes correlations and influence in Python for that timestamp. It does not update the weights.

## 17. How does the frontend receive simulation results?

The simulation panel posts `{"datetime": "..."}` to `POST /api/simulation` through the Vite `/api` proxy. Flask calls `run_simulation` and returns JSON. React draws one node per returned junction and one arrow per returned relationship, and prints the predicted count, historical count, influence index, signed correlation, lag, and contributions from that JSON. It does not recompute the correlation or the sum.

## 18. What are the current limitations and possible future improvements?

Limitations that are true now:

- History ends on 30 June 2017.
- No weather, events, GPS, or road topology.
- Associations are not causes and not roads.
- Spatial GRU’s overall gain on the overlap test is not present at junctions 3 and 4 for RMSE and MAE.
- The two demo hours do not exhibit a negative selected correlation or an excluded source. Those cases are unit-tested with synthetic data.

Proposed only, not implemented:

- Retrain or extend the model if a newer count series is obtained.
- Add weather or events only from a real source.
- Place junctions on a surveyed map only if coordinates exist.
- Train the ARIMA baseline that `ml/README.md` still lists as not trained.
