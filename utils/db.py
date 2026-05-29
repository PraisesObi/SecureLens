"""
SQLite store for SecureLens.
Holds the audit log, case statuses and the local user table.
"""

import os
import json
import sqlite3
import secrets
from datetime import datetime
from typing import Optional, Iterable

from utils.crypto import encrypt_dict, decrypt_dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'securelens.db')


def get_conn() -> sqlite3.Connection:
    """Open a new SQLite connection with row dicts enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables on first run if they do not exist."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT DEFAULT 'analyst',
            created_at    TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       TEXT UNIQUE NOT NULL,
            user_id         INTEGER,
            username        TEXT,
            timestamp       TEXT NOT NULL,
            source          TEXT NOT NULL,
            inputs_json     TEXT NOT NULL,
            label           TEXT NOT NULL,
            confidence      REAL NOT NULL,
            proba_malicious REAL NOT NULL,
            risk_label      TEXT,
            status          TEXT DEFAULT 'open',
            notes           TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
        )
    """)

    # Migration: retroactively promote legacy HIGH RISK rows to CRITICAL RISK
    # so historical data is consistent with the carved-off CVSS-style scheme.
    # Idempotent: only flips rows that still match the old high-confidence
    # malicious pattern (proba_malicious >= 0.90 AND label = MALICIOUS).
    cur.execute(
        """UPDATE predictions
              SET risk_label = 'CRITICAL RISK'
            WHERE label = 'MALICIOUS'
              AND proba_malicious >= 0.90
              AND (risk_label IS NULL OR risk_label NOT LIKE '%CRITICAL%')"""
    )

    # Migration: add model_used column for dual-model audit trail.
    # ALTER TABLE is idempotent-safe: the except block absorbs the
    # 'duplicate column name' error on subsequent startups.
    try:
        cur.execute(
            "ALTER TABLE predictions ADD COLUMN model_used TEXT DEFAULT 'random_forest'"
        )
    except Exception:
        pass

    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str, role: str = 'analyst') -> int:
    """Insert a new user and return its row id."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, role, datetime.utcnow().isoformat())
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def get_user_by_name(username: str) -> Optional[sqlite3.Row]:
    """Find a user by username or return None."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    """Find a user by id or return None."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def create_session(user_id: int) -> str:
    """Create a session token bound to the user."""
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return token


def get_session_user(token: str) -> Optional[sqlite3.Row]:
    """Resolve a session token to a user row or return None."""
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
        (token,)
    ).fetchone()
    conn.close()
    return row


def delete_session(token: str) -> None:
    """Remove a session token to log a user out."""
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def log_prediction(
    report_id: str,
    user_id: Optional[int],
    username: Optional[str],
    source: str,
    inputs: dict,
    label: str,
    confidence: float,
    proba_malicious: float,
    risk_label: str,
    model_used: str = 'random_forest',
) -> None:
    """Encrypt and persist a prediction. MALICIOUS → 'open', NORMAL → 'cleared'."""
    initial_status = 'open' if label == 'MALICIOUS' else 'cleared'
    conn = get_conn()
    conn.execute(
        """INSERT INTO predictions
           (report_id, user_id, username, timestamp, source, inputs_json,
            label, confidence, proba_malicious, risk_label, status, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            report_id,
            user_id,
            username,
            datetime.utcnow().isoformat(),
            source,
            encrypt_dict(inputs),
            label,
            float(confidence),
            float(proba_malicious),
            risk_label,
            initial_status,
            model_used,
        )
    )
    conn.commit()
    conn.close()


def list_predictions(
    label_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 500,
) -> list:
    """Return the audit log rows ordered by newest first."""
    conn = get_conn()
    query = "SELECT * FROM predictions"
    args: list = []
    where = []
    if label_filter:
        where.append("label = ?")
        args.append(label_filter)
    if status_filter:
        where.append("status = ?")
        args.append(status_filter)
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY timestamp DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(query, args).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        # Decrypt inputs back into a dict for the API consumers.
        d['inputs'] = decrypt_dict(d.pop('inputs_json', '') or '')
        out.append(d)
    return out


def update_case_status(report_id: str, status: str, notes: Optional[str] = None) -> bool:
    """Update the case status and optional notes for a prediction row."""
    valid = {'open', 'investigating', 'cleared', 'escalated'}
    if status not in valid:
        return False
    conn = get_conn()
    cur = conn.execute(
        "UPDATE predictions SET status = ?, notes = ? WHERE report_id = ?",
        (status, notes, report_id)
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def get_prediction(report_id: str) -> Optional[dict]:
    """Fetch a single prediction row by report id with inputs decrypted."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM predictions WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['inputs'] = decrypt_dict(d.pop('inputs_json', '') or '')
    return d


def delete_prediction(report_id: str) -> bool:
    """Delete a prediction row by report id. Returns True if a row was removed."""
    conn = get_conn()
    cur = conn.execute("DELETE FROM predictions WHERE report_id = ?", (report_id,))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def bulk_update_status(report_ids: list, status: str, notes: Optional[str] = None) -> int:
    """Update status for a list of report_ids. Returns count of rows changed."""
    valid = {'open', 'investigating', 'cleared', 'escalated'}
    if status not in valid or not report_ids:
        return 0
    conn = get_conn()
    placeholders = ','.join('?' for _ in report_ids)
    cur = conn.execute(
        f"UPDATE predictions SET status = ?, notes = ? WHERE report_id IN ({placeholders})",
        [status, notes] + list(report_ids)
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed


def clear_all_open_cases() -> int:
    """Mark every open/investigating MALICIOUS case as cleared. Returns count changed."""
    conn = get_conn()
    cur = conn.execute(
        "UPDATE predictions SET status = 'cleared' WHERE status IN ('open', 'investigating')"
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed


def get_timeline(days: int = 14) -> dict:
    """Daily malicious/normal counts for the last `days` days (oldest first)."""
    from datetime import timedelta
    today = datetime.utcnow().date()
    buckets = {(today - timedelta(days=i)).isoformat(): {'mal': 0, 'norm': 0}
               for i in range(days)}
    cutoff = (today - timedelta(days=days - 1)).isoformat() + 'T00:00:00'

    conn = get_conn()
    rows = conn.execute(
        "SELECT timestamp, label FROM predictions WHERE timestamp >= ?",
        (cutoff,)
    ).fetchall()
    conn.close()

    for r in rows:
        day = (r['timestamp'] or '')[:10]
        if day in buckets:
            if r['label'] == 'MALICIOUS':
                buckets[day]['mal'] += 1
            else:
                buckets[day]['norm'] += 1

    series = sorted(buckets.items())  # oldest first
    return {
        'days': [d for d, _ in series],
        'malicious': [v['mal']  for _, v in series],
        'normal':    [v['norm'] for _, v in series],
    }


def get_stats() -> dict:
    """Return aggregate counts and risk breakdown for the dashboard banner."""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()['c']
    mal = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE label = 'MALICIOUS'").fetchone()['c']
    norm = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE label = 'NORMAL'").fetchone()['c']
    open_cases = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions WHERE status IN ('open','investigating') AND label='MALICIOUS'"
    ).fetchone()['c']

    # Risk label breakdown for the bar / severity panels on the dashboard.
    risk_rows = conn.execute(
        "SELECT risk_label, COUNT(*) AS c FROM predictions GROUP BY risk_label"
    ).fetchall()
    risk = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for r in risk_rows:
        rl = (r['risk_label'] or '').lower()
        if 'critical' in rl:
            risk['critical'] += r['c']
        elif 'high' in rl:
            risk['high'] += r['c']
        elif 'mod' in rl:
            risk['medium'] += r['c']
        else:
            risk['low'] += r['c']

    conn.close()
    return {
        'total_predictions': total,
        'total_malicious': mal,
        'total_normal': norm,
        'open_cases': open_cases,
        'risk_breakdown': risk,
    }
