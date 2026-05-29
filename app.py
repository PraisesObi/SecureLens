"""SecureLens - FastAPI server: dashboard, predictions, audit log, auth."""

import os
import sys
import io
import json
import uuid
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, ValidationError
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi.exceptions import RequestValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    preprocess_single_record, load_encoders,
    FEATURE_COLS, FEATURE_DISPLAY_NAMES, CATEGORICAL_COLS
)
from utils.explainer import (
    build_explainer, get_shap_values,
    get_feature_importance_df, build_human_readable_explanation,
    get_risk_level
)
from utils.drift import compute_drift_report, load_reference_background
from utils.watcher import start_watcher
from utils import db, auth

MODEL_PATH      = os.path.join('model', 'rf_model.joblib')
LR_PATH         = os.path.join('model', 'lr_model.joblib')
SCALER_PATH     = os.path.join('model', 'lr_scaler.joblib')
ENCODER_PATH    = os.path.join('model', 'encoders.joblib')
BG_DATA_PATH    = os.path.join('model', 'shap_background.joblib')
UNIQUE_PATH     = os.path.join('model', 'unique_values.joblib')
METRICS_PATH    = os.path.join('model', 'model_comparison.json')
TEMPLATE_PATH   = os.path.join('templates', 'index.html')

print("Loading model artifacts ...")
try:
    rf_model     = joblib.load(MODEL_PATH)
    encoders     = load_encoders(ENCODER_PATH)
    bg_data      = joblib.load(BG_DATA_PATH)
    unique_vals  = joblib.load(UNIQUE_PATH)
    explainer    = build_explainer(rf_model, bg_data)
    try:
        lr_model  = joblib.load(LR_PATH)
        lr_scaler = joblib.load(SCALER_PATH)
    except Exception:
        lr_model  = None
        lr_scaler = None
    # Load the recall-optimised decision threshold produced by train_model.py.
    # Falls back to 0.5 when the JSON is absent (e.g. first run before training).
    _DEFAULT_THRESHOLD: float = 0.5
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as _f:
                _DEFAULT_THRESHOLD = float(json.load(_f).get('optimal_threshold', 0.5))
        except Exception:
            pass
    print(f"All artifacts loaded. Default threshold: {_DEFAULT_THRESHOLD}")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Run 'python train_model.py' first.")
    rf_model = encoders = bg_data = unique_vals = explainer = None
    lr_model = lr_scaler = None
    _DEFAULT_THRESHOLD = 0.5

auth.ensure_default_admin()

# Rate-limited per IP: login 10/min, predictions 60/min, batch 10/min.
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(title="SecureLens", version="2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors to readable 400 responses."""
    messages = []
    for err in exc.errors():
        field = " → ".join(str(x) for x in err.get("loc", []) if x != "body")
        messages.append(f"{field}: {err['msg']}" if field else err['msg'])
    detail = "; ".join(messages) if messages else "Invalid input"
    return JSONResponse(status_code=400, content={"detail": detail})


# Background folder watcher for proactive auto detection of new CSV drops.
_watcher_observer = None


@app.on_event("startup")
def _spin_up_watcher() -> None:
    """Start background folder watcher for auto CSV ingestion."""
    global _watcher_observer
    if rf_model is None:
        return
    try:
        # Wrap _run_prediction so the watcher uses the recall-optimised threshold
        # (_DEFAULT_THRESHOLD) rather than the hardcoded 0.5 inside watcher.py.
        # The watcher always uses RF (the primary model) for autonomous scoring.
        _watcher_observer = start_watcher(
            lambda rec, _t: _run_prediction(rec, _DEFAULT_THRESHOLD, model_key='rf'),
            db.log_prediction,
        )
    except Exception as e:
        print(f"[watcher] Failed to start: {e}")


@app.on_event("shutdown")
def _shut_down_watcher() -> None:
    """Stop folder watcher cleanly."""
    global _watcher_observer
    if _watcher_observer is not None:
        try:
            _watcher_observer.stop()
            _watcher_observer.join(timeout=2)
        except Exception:
            pass


class EmployeeRecord(BaseModel):
    """Schema for a single employee record submitted from the form."""
    employee_department: str
    employee_campus: str
    employee_position: str
    employee_seniority_years: float
    is_contractor: int
    employee_classification: int
    has_foreign_citizenship: int
    has_criminal_record: int
    has_medical_history: int
    employee_origin_country: str
    total_printed_pages: float
    num_printed_pages_off_hours: float
    total_files_burned: float
    burned_from_other: float
    is_abroad: int
    trip_day_number: float
    hostility_country_level: float
    num_entries: float
    num_unique_campus: float
    late_exit_flag: int
    entry_during_weekend: int
    threshold: Optional[float] = None  # resolved to _DEFAULT_THRESHOLD at request time
    model: Optional[str] = 'rf'         # 'rf' (Random Forest) or 'lr' (Logistic Regression)

    @validator('model')
    def valid_model(cls, v: Optional[str]) -> str:
        allowed = {'rf', 'lr'}
        if v not in (allowed | {None}):
            raise ValueError(f'Model must be one of: rf, lr')
        return v or 'rf'

    @validator('employee_department', 'employee_campus', 'employee_position', 'employee_origin_country')
    def no_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Field must not be empty')
        return v.strip()

    @validator('is_contractor', 'has_foreign_citizenship', 'has_criminal_record',
               'has_medical_history', 'is_abroad', 'late_exit_flag', 'entry_during_weekend')
    def binary_flag(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError('Flag fields must be 0 or 1')
        return v

    @validator('employee_classification')
    def clearance_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError('Classification level must be between 1 and 5')
        return v

    @validator('hostility_country_level')
    def hostility_range(cls, v: float) -> float:
        if not (0 <= v <= 5):
            raise ValueError('Hostility level must be between 0 and 5')
        return v

    @validator('employee_seniority_years', 'total_printed_pages', 'num_printed_pages_off_hours',
               'total_files_burned', 'burned_from_other', 'trip_day_number', 'num_entries', 'num_unique_campus')
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Numeric field must not be negative')
        return v

    @validator('threshold', pre=True, always=True)
    def threshold_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= float(v) <= 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
        return v


class LoginPayload(BaseModel):
    """Schema for the login form."""
    username: str
    password: str


class StatusPayload(BaseModel):
    """Schema for updating a case status."""
    status: str
    notes: Optional[str] = None

    @validator('status')
    def valid_status(cls, v: str) -> str:
        allowed = {'open', 'investigating', 'cleared', 'escalated'}
        if v not in allowed:
            raise ValueError(f'Status must be one of: {", ".join(sorted(allowed))}')
        return v

    @validator('notes')
    def notes_length(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > 500:
            raise ValueError('Notes must not exceed 500 characters')
        return v


class BulkStatusPayload(BaseModel):
    """Schema for bulk-updating case statuses."""
    report_ids: list
    status: str
    notes: Optional[str] = None

    @validator('status')
    def valid_status(cls, v: str) -> str:
        allowed = {'open', 'investigating', 'cleared', 'escalated'}
        if v not in allowed:
            raise ValueError(f'Status must be one of: {", ".join(sorted(allowed))}')
        return v

    @validator('report_ids')
    def non_empty_ids(cls, v: list) -> list:
        if not v:
            raise ValueError('report_ids must not be empty')
        if len(v) > 500:
            raise ValueError('Cannot update more than 500 records at once')
        return v


def _current_user(authorization: Optional[str]) -> Optional[dict]:
    """Resolve Bearer token to a user dict, or None."""
    if not authorization:
        return None
    token = authorization.replace('Bearer ', '').strip()
    return auth.user_from_token(token)


def _require_user(authorization: Optional[str]) -> dict:
    """Return current user or raise 401."""
    user = _current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _run_prediction(record_dict: dict, threshold: float = 0.5,
                    model_key: str = 'rf') -> dict:
    """Score one record with the chosen model and return the result payload.

    model_key='rf'  → Random Forest with SHAP TreeExplainer (primary model)
    model_key='lr'  → Logistic Regression with coefficient-based feature
                      importance using the same fi_df schema so the UI chart
                      renders identically for both models.
    """
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    X = preprocess_single_record(record_dict, encoders)

    if model_key == 'lr':
        if lr_model is None or lr_scaler is None:
            raise HTTPException(status_code=503,
                                detail="Logistic Regression model not available. Run train_model.py first.")
        X_scaled   = lr_scaler.transform(X)
        proba      = lr_model.predict_proba(X_scaled)[0]
        pred_class = 1 if proba[1] >= threshold else 0
        confidence = proba[pred_class]
        # Build fi_df from raw LR coefficients (same schema as SHAP output).
        # Positive coef → feature pushes prediction toward MALICIOUS (direction='risk').
        coefs = lr_model.coef_[0]
        fi_df = pd.DataFrame({
            'feature':      FEATURE_COLS,
            'shap_value':   coefs,                                      # signed coefficient
        })
        fi_df['display_name'] = fi_df['feature'].map(
            lambda f: FEATURE_DISPLAY_NAMES.get(f, f)
        )
        fi_df['abs_impact'] = fi_df['shap_value'].abs()
        fi_df['direction']  = fi_df['shap_value'].apply(
            lambda v: 'risk' if v > 0 else 'safe'
        )
        fi_df = fi_df.sort_values('abs_impact', ascending=False).reset_index(drop=True)
        shap_available = False
    else:
        proba      = rf_model.predict_proba(X)[0]
        pred_class = 1 if proba[1] >= threshold else 0
        confidence = proba[pred_class]
        shap_vals, _  = get_shap_values(explainer, X.values)
        fi_df         = get_feature_importance_df(shap_vals, FEATURE_COLS)
        shap_available = True

    label         = 'MALICIOUS' if pred_class == 1 else 'NORMAL'
    explanation   = build_human_readable_explanation(label, confidence, fi_df, record_dict)
    risk_label, _ = get_risk_level(confidence, pred_class)
    top_features  = fi_df.head(8).to_dict(orient='records')

    return {
        'label':           label,
        'pred_class':      int(pred_class),
        'confidence':      round(float(confidence), 4),
        'proba_normal':    round(float(proba[0]), 4),
        'proba_malicious': round(float(proba[1]), 4),
        'risk_label':      risk_label,
        'explanation':     explanation,
        'top_features':    top_features,
        'model_used':      'logistic_regression' if model_key == 'lr' else 'random_forest',
        'shap_available':  shap_available,
    }


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the SPA shell."""
    if not os.path.exists(TEMPLATE_PATH):
        return HTMLResponse("<h1>Template not found</h1>", status_code=500)
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())


@app.post("/api/auth/login")
@limiter.limit("10/minute")
def auth_login(request: Request, payload: LoginPayload):
    """Validate credentials and return a session token (rate-limited: 10/min)."""
    token = auth.login(payload.username, payload.password)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user = auth.user_from_token(token)
    return {'token': token, 'user': user}


@app.post("/api/auth/logout")
def auth_logout(authorization: Optional[str] = Header(None)):
    """Invalidate the current session token."""
    if authorization:
        auth.logout(authorization.replace('Bearer ', '').strip())
    return {'ok': True}


@app.get("/api/auth/me")
def auth_me(authorization: Optional[str] = Header(None)):
    """Return the current user or 401."""
    user = _current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {'user': user}


@app.get("/api/unique-values")
def get_unique_values_endpoint(authorization: Optional[str] = Header(None)):
    """Return categorical values for form dropdowns."""
    _require_user(authorization)
    if unique_vals is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {k: [str(v) for v in vals] for k, vals in unique_vals.items()}


@app.post("/api/predict")
@limiter.limit("60/minute")
def predict_single(request: Request, record: EmployeeRecord, authorization: Optional[str] = Header(None)):
    """Score a single employee record and log to audit (rate-limited: 60/min)."""
    user = _require_user(authorization)
    data      = record.dict()
    model_key = data.pop('model', 'rf') or 'rf'
    threshold = data.pop('threshold', None)
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD
    result = _run_prediction(data, threshold, model_key=model_key)

    report_id = str(uuid.uuid4())
    db.log_prediction(
        report_id=report_id,
        user_id=user['id'],
        username=user['username'],
        source='single',
        inputs=data,
        label=result['label'],
        confidence=result['confidence'],
        proba_malicious=result['proba_malicious'],
        risk_label=result['risk_label'],
        model_used=result.get('model_used', 'random_forest'),
    )

    result['report_id'] = report_id
    return JSONResponse(result)


@app.post("/api/predict-batch")
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = 0.5,
    model: str = 'rf',
    authorization: Optional[str] = Header(None),
):
    """Score a CSV and return the threat leaderboard (rate-limited: 10/min)."""
    user = _require_user(authorization)
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file contains no data rows.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    results = []
    for i, row in df.iterrows():
        record_dict = row[FEATURE_COLS].to_dict()
        try:
            r = _run_prediction(record_dict, threshold, model_key=model)
            report_id = str(uuid.uuid4())
            db.log_prediction(
                report_id=report_id,
                user_id=user['id'],
                username=user['username'],
                source='batch',
                inputs=record_dict,
                label=r['label'],
                confidence=r['confidence'],
                proba_malicious=r['proba_malicious'],
                risk_label=r['risk_label'],
                model_used=r.get('model_used', 'random_forest'),
            )

            results.append({
                'row':             int(i) + 1,
                'report_id':       report_id,
                'department':      record_dict.get('employee_department', ''),
                'position':        record_dict.get('employee_position', ''),
                'campus':          record_dict.get('employee_campus', ''),
                'label':           r['label'],
                'risk_label':      r['risk_label'],
                'confidence':      r['confidence'],
                'proba_malicious': r['proba_malicious'],
            })
        except Exception as e:
            results.append({'row': int(i) + 1, 'error': str(e)})

    results.sort(key=lambda x: x.get('proba_malicious', 0), reverse=True)

    summary = {
        'total':     len(results),
        'malicious': sum(1 for r in results if r.get('label') == 'MALICIOUS'),
        'normal':    sum(1 for r in results if r.get('label') == 'NORMAL'),
    }

    return JSONResponse({'summary': summary, 'results': results})


@app.get("/api/model-comparison")
def model_comparison(authorization: Optional[str] = Header(None)):
    """Return precomputed RF vs LR metrics."""
    _require_user(authorization)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return JSONResponse(json.load(f))
    raise HTTPException(status_code=404, detail="Run train_model.py to generate comparison data.")


@app.get("/api/history")
def history(
    label: Optional[str] = None,
    status: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """Return audit log rows, optionally filtered by label or status."""
    _require_user(authorization)
    rows = db.list_predictions(label_filter=label, status_filter=status)
    return {'rows': rows}


@app.patch("/api/history/{report_id}/status")
def update_status(
    report_id: str,
    payload: StatusPayload,
    authorization: Optional[str] = Header(None),
):
    """Update case status and optional notes for a prediction."""
    _require_user(authorization)
    ok = db.update_case_status(report_id, payload.status, payload.notes)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid status or report id")
    return {'ok': True}


@app.delete("/api/history/{report_id}")
def delete_history(
    report_id: str,
    authorization: Optional[str] = Header(None),
):
    """Permanently delete a prediction record from the audit log."""
    _require_user(authorization)
    ok = db.delete_prediction(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found")
    return {'ok': True}


@app.patch("/api/history/bulk-status")
def bulk_status(
    payload: BulkStatusPayload,
    authorization: Optional[str] = Header(None),
):
    """Bulk-update status for multiple prediction records."""
    _require_user(authorization)
    count = db.bulk_update_status(payload.report_ids, payload.status, payload.notes)
    return {'ok': True, 'updated': count}


@app.post("/api/history/clear-all")
def clear_all_open(authorization: Optional[str] = Header(None)):
    """Mark all open/investigating cases as cleared."""
    _require_user(authorization)
    count = db.clear_all_open_cases()
    return {'ok': True, 'cleared': count}


@app.get("/api/stats")
def stats(authorization: Optional[str] = Header(None)):
    """Return aggregate counts and risk breakdown for the dashboard."""
    _require_user(authorization)
    return db.get_stats()


@app.get("/api/timeline")
def timeline(days: int = 14, authorization: Optional[str] = Header(None)):
    """Return daily prediction counts (malicious vs normal) for the last N days."""
    _require_user(authorization)
    days = max(1, min(int(days), 60))  # cap to 1..60 days
    return db.get_timeline(days)


@app.get("/api/drift")
def drift_report(authorization: Optional[str] = Header(None)):
    """Return PSI drift report comparing training data to recent predictions."""
    _require_user(authorization)
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    reference_df = load_reference_background(BG_DATA_PATH)
    if reference_df is None or reference_df.empty:
        raise HTTPException(status_code=503, detail="Reference background not available.")

    rows = db.list_predictions(limit=500)
    if not rows:
        return JSONResponse({
            'features':       [],
            'max_psi':        0.0,
            'overall_status': 'stable',
            'reference_n':    int(len(reference_df)),
            'current_n':      0,
            'message':        'No predictions logged yet. Run a few predictions to enable drift detection.',
        })

    encoded_rows = []
    for r in rows:
        inputs = r.get('inputs') or {}
        if not inputs:
            continue
        try:
            encoded_rows.append(preprocess_single_record(inputs, encoders).iloc[0])
        except Exception:
            continue

    if not encoded_rows:
        raise HTTPException(status_code=503, detail="Could not encode logged records.")

    current_df = pd.DataFrame(encoded_rows, columns=FEATURE_COLS)
    report = compute_drift_report(reference_df, current_df)
    return JSONResponse(report)




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=False)
