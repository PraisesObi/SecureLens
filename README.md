# SecureLens

**Insider Threat Detection System** built around a dual-model engine (Random Forest primary + Logistic Regression baseline) with SHAP / coefficient explainability, a local audit log and case management, and an authenticated dashboard.

This iteration is the v2 modernisation of the original COS720 prototype. It adds a splash screen, a login flow with bcrypt-hashed credentials, a SQLite audit trail, history filtering, case status workflow, an animated tunnel background, a glass-style dashboard, and a click-to-enlarge panel system.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)

```bash
python train_model.py
```

This trains and compares **Random Forest** against **Logistic Regression**, then writes all artefacts under `model/`.

### 3. Launch the app

On Windows, just double-click **`launch.bat`** (or **`SecureLens.url`** if the server is already running). It starts the FastAPI server on `http://127.0.0.1:8000` and opens the browser.

Or run it directly:

```bash
python app.py
```

### 4. Sign in

Default credentials on first run:

- **Username:** `admin`
- **Password:** `admin123`

The first launch seeds this account into `securelens.db` with a bcrypt hash. Update the password through the database (or replace the hash inside `utils/auth.py:DEFAULT_PASSWORD` and delete `securelens.db`) before any real use.

---

## Features

| Feature                | Description                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Splash + Login         | Animated SecureLens scan logo, then a glass-style login over the tunnel background                                   |
| Dashboard              | Glass panels over a live canvas tunnel, animated counters, click-to-enlarge tiles                                    |
| Single Analysis        | Form-based classification with SHAP feature contributions and analyst explanation                                    |
| **Model Selection**    | Toggle between Random Forest (SHAP explanations) and Logistic Regression (coefficient explanations) on any prediction |
| Batch Upload           | Drag-and-drop CSV scoring with a sortable risk leaderboard — model selector applies to bulk runs too                 |
| **Folder Watcher**     | Drop a CSV into `watched/` and it is auto-scored, audit-logged and archived without any user action                  |
| Audit History          | Every prediction is persisted to SQLite with timestamp, analyst and inputs                                           |
| Case Management        | Mark a flagged record as Open, Investigating, Cleared or Escalated                                                   |
| Model Comparison       | RF versus LR breakdown with confusion matrix and rationale                                                           |
| Sensitivity Slider     | Adjust the malicious threshold in real time                                                                          |
| **Input Validation**   | Frontend range/type checks on every field; backend Pydantic validators return clear 400 errors                       |
| Local Auth             | bcrypt-hashed credentials, token sessions stored in SQLite                                                           |
| **Encryption at Rest** | Behavioural records are encrypted with Fernet (AES-128 CBC + HMAC-SHA256) before they touch SQLite                   |
| **Rate Limiting**      | Per-IP throttling on login (10/min), single predict (60/min) and batch (10/min) for brute force and abuse prevention |
| **Test Suite**         | 10 pytest tests producing reproducible TP/FP/TN/FN counts plus failure case dumps                                    |
| **Drift Detection**    | Population Stability Index per feature comparing training distribution vs runtime traffic                            |
| **Threat Timeline**    | 14-day stacked bar chart of malicious vs normal predictions on the dashboard; click to enlarge                       |
| **Confusion Matrix**   | Live RF/LR confusion matrix on the Models tab with miss-rate and false-alarm-rate summary                            |
| **Risk Score Breakdown** | Per-prediction decomposition showing each feature's percentage contribution to the risk score                      |
| **Audit CSV Export**   | Export the current filtered audit history to CSV for external analysis or hand-off                                  |

---

## Project Structure

```
insider_threat/
├── app.py                  # FastAPI server (run this)
├── launch.bat              # Windows launcher (server + browser)
├── restart.bat             # Kills port 8000 and relaunches the server
├── test.csv                # Sample 10-row CSV for batch upload testing
├── SecureLens.url          # One-click shortcut to the local URL
├── securelens.json         # App manifest with endpoints and metadata
├── train_model.py          # Model training pipeline
├── requirements.txt
├── README.md
├── data/
│   └── insider_threat_clean_dataset.csv
├── model/
│   ├── rf_model.joblib     # Random Forest (primary)
│   ├── lr_model.joblib     # Logistic Regression (baseline)
│   ├── encoders.joblib     # Categorical encoders
│   ├── shap_background.joblib
│   ├── unique_values.joblib
│   └── model_comparison.json
├── templates/
│   └── index.html          # Single page application shell
├── tests/                  # pytest suite. Run: python -m pytest tests/ -v
│   ├── conftest.py         # Shared fixtures (model, encoders, sample)
│   ├── test_predictions.py # TP/FP/TN/FN harness, writes results.json
│   ├── test_drift.py       # PSI sanity checks
│   └── test_crypto.py      # Encryption round trip tests
├── watched/                # Drop CSVs here for auto scoring (created on boot)
│   ├── processed/          # Successfully scored CSVs land here
│   └── failed/             # Malformed CSVs are quarantined here
└── utils/
    ├── preprocessing.py    # Feature encoding helpers
    ├── explainer.py        # SHAP explanations
    ├── db.py               # SQLite store (audit log, users, sessions)
    ├── auth.py             # bcrypt password hashing and login flow
    ├── crypto.py           # Fernet at-rest encryption for the audit log
    ├── drift.py            # Population Stability Index drift detection
    └── watcher.py          # Background folder watcher for proactive detection
```

`securelens.db` is created automatically on first run and lives next to `app.py`. Delete it to reset users and audit log.

---

## API Surface

| Method | Path                              | Purpose                                                        |
| ------ | --------------------------------- | -------------------------------------------------------------- |
| GET    | `/`                               | Serves the SPA                                                 |
| POST   | `/api/auth/login`                 | Sign in with username and password                             |
| POST   | `/api/auth/logout`                | Invalidate the current session                                 |
| GET    | `/api/auth/me`                    | Validate the bearer token                                      |
| GET    | `/api/unique-values`              | Categorical options for form dropdowns                         |
| POST   | `/api/predict`                    | Score a single record (`model`: `rf`\|`lr`, default `rf`)       |
| POST   | `/api/predict-batch`              | Score a CSV (`?model=rf\|lr&threshold=0.5`), return leaderboard |
| GET    | `/api/history`                    | Audit log filtered by label or status                          |
| PATCH  | `/api/history/{report_id}/status` | Set case status                                                |
| DELETE | `/api/history/{report_id}`        | Permanently delete a prediction record from the audit log      |
| PATCH  | `/api/history/bulk-status`        | Update status for multiple records in one request              |
| POST   | `/api/history/clear-all`          | Mark every open and investigating case as cleared              |
| GET    | `/api/stats`                      | Aggregate counts for the dashboard                             |
| GET    | `/api/timeline`                   | Daily malicious/normal counts for the last N days              |
| GET    | `/api/model-comparison`           | RF vs LR metrics                                               |
| GET    | `/api/drift`                      | Population Stability Index per feature plus an overall verdict |

All `/api/*` routes except the login require a `Authorization: Bearer <token>` header.

---

## Model Performance

### Random Forest (primary) — threshold = 0.50

| Metric     | Value  |
| ---------- | ------ |
| Accuracy   | ~97%   |
| Precision  | ~68%   |
| **Recall** | **~82%** |
| F1-Score   | ~75%   |

### Random Forest — optimal threshold (recall ≥ 85%)

After training, `train_model.py` sweeps the precision-recall curve to find the lowest decision threshold at which recall ≥ 85% while maximising F1. This threshold is saved in `model/model_comparison.json` as `optimal_threshold` and becomes the server default at startup.

| Metric     | Value (re-run `train_model.py` to populate) |
| ---------- | ------------------------------------------- |
| Threshold  | see `model/model_comparison.json`           |
| **Recall** | **≥ 85%** (guaranteed by optimisation)     |
| F1-Score   | maximised subject to recall floor          |

### Logistic Regression (comparison baseline)

| Metric     | Value  |
| ---------- | ------ |
| Accuracy   | ~86%   |
| Precision  | ~23%   |
| **Recall** | **~71%** |
| F1-Score   | ~35%   |

Recall is the critical metric. A false negative means a real threat went undetected. RF is selected as the primary model because its recall, precision and F1 all dominate LR, and its non-linear decision boundary captures complex insider-threat patterns that LR's linear boundary cannot.

---

## Dataset

- 118,614 employee behaviour records
- 21 features (profile, behaviour, access patterns)
- Class imbalance: 94.6% Normal / 5.4% Malicious
- SMOTE applied to balance training data

---

## Running the Test Suite

```bash
python -m pytest tests/ -v
```

Produces:

- `tests/results.json` — confusion counts and metrics at the **optimal threshold** on a balanced 50/50 stratified sample. The `threshold_source` field records whether this is the optimised or default threshold.
- `tests/failures.json` — every false positive and false negative, each tagged with a `root_cause` label explaining the failure pattern (ambiguous signal, domain mismatch, feature co-occurrence, etc.).

Note: the stratified sample intentionally over-represents the malicious class (50/50 vs ~5% natural prevalence) for test convenience. Full-dataset metrics at both thresholds are in `model/model_comparison.json`.

---

## Security Notes

- **At-rest encryption:** the local Fernet key lives at `model/securelens.key`. Lose it and historical predictions cannot be decrypted, so back it up if you ever depend on the audit history.
- **Rate limits** are stored in process memory. Restarting the server resets the counters. If you scale beyond a single process, point slowapi at Redis.
- **Default admin account** must be rotated before any production exposure. Edit the bcrypt hash in `utils/auth.py:DEFAULT_PASSWORD` and delete `securelens.db` to reseed.
- **Watcher folder** is local filesystem only. If this tool ever ingests from a SIEM directly, replace the `watchdog` observer with a streaming consumer (Kafka, Kinesis, etc.).

---

## Known Limitations

These are deliberate scope boundaries, not bugs. They are noted in the report's Limitations section:

- **No per-employee behavioural baseline.** Each record is classified independently against a population-trained model. There is no longitudinal "John usually prints 5 pages, today he printed 200" comparison. Adding per-user baselines would address the four-phase DFR model proposed by Shoderu et al. (2025) and is a clear direction for future work.
- **Audit log is encrypted but not hash-chained.** Fernet provides per-row confidentiality and integrity, but rows are independent. An admin with DB access could delete a row and the remaining rows would still validate. True forensic chain-of-custody requires a Merkle/`prev_hash` chain over the audit table.

---

## Suggested Next Steps

1. **Per-user accounts.** Add a small admin screen to create extra analysts so audit rows can be attributed to individuals.
2. **Notes per case.** The DB already supports it. A text area in the history row would close that loop.
3. **Hash-chained audit log.** Add a `prev_hash` column over the encrypted blob so tampering or row deletion is mathematically detectable.
4. **Per-employee behavioural baseline.** Track each user's rolling history so deviations from their own pattern flag alongside population anomalies (the Shoderu et al. four-phase DFR model).
5. **Drift dashboard tile.** Surface the `/api/drift` verdict as a visible card on the dashboard.
6. **Live SIEM ingestion.** Swap the folder watcher for a Kafka or webhook consumer if you ever go beyond local files.

The architecture supports each of these without touching the model layer.
