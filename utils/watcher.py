"""
Real time CSV watcher for SecureLens.

Continuously monitors the `watched/` folder. Whenever a new `.csv` file lands
there it is automatically scored through the prediction pipeline and every row
is written to the audit log under the system 'auto-watcher' user. After
processing the file is moved into `watched/processed/` so it is not picked up
again on restart.

This makes the system proactive rather than purely reactive: an upstream SIEM
or log forwarder can drop a CSV into the watched folder and SecureLens will
detect, score and audit it without any human action.
"""

import os
import shutil
import time
import threading
import uuid
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils.preprocessing import FEATURE_COLS

WATCHED_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'watched')
PROCESSED_DIR = os.path.join(WATCHED_DIR, 'processed')
FAILED_DIR    = os.path.join(WATCHED_DIR, 'failed')


def _ensure_dirs() -> None:
    """Create the watched folder layout on first use."""
    for d in (WATCHED_DIR, PROCESSED_DIR, FAILED_DIR):
        os.makedirs(d, exist_ok=True)


class _CsvHandler(FileSystemEventHandler):
    """Reacts to CSV files appearing inside the watched folder."""

    def __init__(self, on_csv: Callable[[str], None]):
        super().__init__()
        self.on_csv = on_csv

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith('.csv'):
            return
        # Wait briefly so the writer can finish flushing the file to disk.
        time.sleep(0.5)
        self.on_csv(event.src_path)


def _process_csv(
    csv_path: str,
    run_prediction: Callable[[dict, float], dict],
    log_prediction: Callable[..., None],
) -> dict:
    """Score every row in `csv_path` and audit log the results.

    Returns a small summary dict. Moves the file into `processed/` on success,
    or `failed/` if it could not be parsed.
    """
    fname = os.path.basename(csv_path)
    summary = {'file': fname, 'total': 0, 'malicious': 0, 'normal': 0, 'errors': 0}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        shutil.move(csv_path, os.path.join(FAILED_DIR, fname))
        summary['errors'] = -1
        summary['error_message'] = f'Could not parse CSV: {e}'
        return summary

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        shutil.move(csv_path, os.path.join(FAILED_DIR, fname))
        summary['errors'] = -1
        summary['error_message'] = f'Missing columns: {missing}'
        return summary

    for _, row in df.iterrows():
        record_dict = row[FEATURE_COLS].to_dict()
        try:
            r = run_prediction(record_dict, 0.5)
            report_id = str(uuid.uuid4())
            log_prediction(
                report_id=report_id,
                user_id=None,
                username='auto-watcher',
                source='watched',
                inputs=record_dict,
                label=r['label'],
                confidence=r['confidence'],
                proba_malicious=r['proba_malicious'],
                risk_label=r['risk_label'],
            )
            summary['total'] += 1
            if r['label'] == 'MALICIOUS':
                summary['malicious'] += 1
            else:
                summary['normal'] += 1
        except Exception:
            summary['errors'] += 1

    # Stamp the destination filename with a timestamp so reruns do not collide.
    dest_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{fname}"
    shutil.move(csv_path, os.path.join(PROCESSED_DIR, dest_name))
    return summary


def start_watcher(
    run_prediction: Callable[[dict, float], dict],
    log_prediction: Callable[..., None],
) -> Optional[Observer]:
    """Start the folder watcher in a background thread.

    Returns the running Observer so the caller can stop it on shutdown.
    """
    _ensure_dirs()

    def _on_csv(path: str) -> None:
        # Run the processor in a worker thread so the watchdog event loop is
        # never blocked by long predictions.
        threading.Thread(
            target=lambda: _log_summary(_process_csv(path, run_prediction, log_prediction)),
            daemon=True,
        ).start()

    handler = _CsvHandler(_on_csv)
    observer = Observer()
    observer.schedule(handler, WATCHED_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[watcher] Watching '{WATCHED_DIR}' for new CSV files.")
    return observer


def _log_summary(summary: dict) -> None:
    """Print a single line summary after a CSV has been processed."""
    if summary.get('errors', 0) == -1:
        print(f"[watcher] FAIL {summary['file']}: {summary.get('error_message', 'unknown error')}")
        return
    print(
        f"[watcher] OK {summary['file']}: "
        f"{summary['total']} scored, "
        f"{summary['malicious']} malicious, "
        f"{summary['normal']} normal, "
        f"{summary['errors']} errors"
    )
