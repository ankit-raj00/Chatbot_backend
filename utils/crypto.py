"""
Symmetric encryption for secrets at rest (MCP server headers, OAuth tokens/client
credentials). Uses Fernet (AES-128-CBC + HMAC) with a key derived from
JWT_SECRET_KEY via HKDF, so no separate encryption key needs to be provisioned —
but the derived key is cryptographically independent of the JWT signing use.
"""
import base64
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY must be set to derive the MCP secrets encryption key")
    key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"agentx-mcp-secrets-encryption-v1",
    ).derive(secret.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(key_material))


def encrypt_str(plaintext: str) -> str:
    """Encrypt a string, returning a token safe to store in MongoDB."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_str(token: str) -> str:
    """Decrypt a value previously produced by encrypt_str. Raises ValueError on
    tampered/invalid ciphertext (e.g. key rotated, or plaintext accidentally
    passed in) rather than leaking cryptography-internal exception types."""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Could not decrypt value — invalid token or wrong key") from e


def encrypt_dict_values(d: dict[str, str] | None) -> dict[str, str] | None:
    """Encrypt every value in a flat str->str dict (e.g. custom headers)."""
    if d is None:
        return None
    return {k: encrypt_str(v) for k, v in d.items()}


def decrypt_dict_values(d: dict[str, str] | None) -> dict[str, str] | None:
    """Decrypt every value in a flat str->str dict."""
    if d is None:
        return None
    return {k: decrypt_str(v) for k, v in d.items()}
