"""
Auth helpers for SecureLens.
Wraps bcrypt for password hashing and bridges sessions through the SQLite store.
"""

from typing import Optional
import bcrypt

from utils import db

DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin123'


def hash_password(plain: str) -> str:
    """Return a bcrypt hash for a plain text password."""
    return bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain text password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


def ensure_default_admin() -> None:
    """Seed a default admin user on first run if no admin exists."""
    db.init_db()
    existing = db.get_user_by_name(DEFAULT_USERNAME)
    if existing is None:
        db.create_user(
            username=DEFAULT_USERNAME,
            password_hash=hash_password(DEFAULT_PASSWORD),
            role='admin',
        )


def login(username: str, password: str) -> Optional[str]:
    """Validate credentials and return a new session token on success."""
    user = db.get_user_by_name(username)
    if user is None:
        return None
    if not verify_password(password, user['password_hash']):
        return None
    return db.create_session(user['id'])


def logout(token: str) -> None:
    """Invalidate a session token."""
    db.delete_session(token)


def user_from_token(token: Optional[str]) -> Optional[dict]:
    """Resolve a session token to a user dict or return None."""
    row = db.get_session_user(token)
    if row is None:
        return None
    return {'id': row['id'], 'username': row['username'], 'role': row['role']}
