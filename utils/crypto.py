"""
At-rest encryption helpers for SecureLens.

Employee behavioural records contain sensitive information (department,
position, criminal history flags, citizenship). They are stored inside the
local SQLite audit log and must therefore be encrypted at rest.

The key is generated on first run and stored in `model/securelens.key` with
restrictive permissions. The encryption uses Fernet (AES 128 in CBC mode with
HMAC SHA 256 authentication) which is the high level symmetric primitive
recommended by the cryptography library maintainers.
"""

import json
import os
import stat
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

KEY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model', 'securelens.key'
)

_fernet: Optional[Fernet] = None


def _ensure_key() -> bytes:
    """Load the local key, or generate a new one on first use."""
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, 'rb') as f:
            return f.read().strip()

    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    key = Fernet.generate_key()
    with open(KEY_PATH, 'wb') as f:
        f.write(key)
    # Lock down permissions on POSIX. On Windows os.chmod is mostly a no op
    # but we still call it for a clear best effort intent.
    try:
        os.chmod(KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return key


def _get_fernet() -> Fernet:
    """Lazy load the Fernet instance the first time it is needed."""
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_ensure_key())
    return _fernet


def encrypt_dict(data: dict) -> str:
    """Encrypt a JSON serialisable dict and return a string token."""
    raw = json.dumps(data, default=str).encode('utf-8')
    token = _get_fernet().encrypt(raw)
    return token.decode('utf-8')


def decrypt_dict(token: str) -> dict:
    """Decrypt a token back into a dict.

    If the token is not a valid Fernet token we assume it is legacy plaintext
    JSON (rows that were written before encryption was enabled) and return it
    parsed. This keeps the audit log readable across the upgrade boundary.
    """
    if not token:
        return {}
    try:
        raw = _get_fernet().decrypt(token.encode('utf-8'))
        return json.loads(raw.decode('utf-8'))
    except InvalidToken:
        try:
            return json.loads(token)
        except Exception:
            return {}
    except Exception:
        return {}
