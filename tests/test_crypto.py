"""
Round trip tests for the at rest encryption helpers. Confirms that a record
can be encrypted and recovered without loss, and that a malformed token
falls back gracefully without crashing the audit log read path.
"""

from utils.crypto import encrypt_dict, decrypt_dict


def test_round_trip_preserves_dict():
    payload = {
        'employee_department': 'Finance',
        'is_contractor': 0,
        'total_files_burned': 12.0,
    }
    token = encrypt_dict(payload)
    assert isinstance(token, str)
    assert token != ''
    recovered = decrypt_dict(token)
    assert recovered == payload


def test_decrypt_legacy_plain_json_still_works():
    """Rows written before encryption was switched on are stored as raw JSON.
    The decrypt helper must read them transparently for backwards compatibility.
    """
    legacy = '{"employee_department": "IT", "is_contractor": 1}'
    recovered = decrypt_dict(legacy)
    assert recovered == {'employee_department': 'IT', 'is_contractor': 1}


def test_decrypt_garbage_returns_empty_dict():
    """A corrupt token must not raise, so the history endpoint stays robust."""
    assert decrypt_dict('not-a-valid-token') == {}
    assert decrypt_dict('') == {}
