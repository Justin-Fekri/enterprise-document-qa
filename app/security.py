"""Authentication, password hashing and role-based document permissions.

Permissions are expressed once, here, as a role -> (categories, sensitivity
ceiling) table. `app.retrieval.retriever.user_can_read` is the only place that
evaluates them, and it is applied during retrieval - so an unreadable passage
is never retrieved, never sent to the model, and never citable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt

from app.config import Settings
from app.schemas import DocumentCategory, Role, Sensitivity, User

PBKDF2_ITERATIONS = 240_000

# Role -> what that role may read. `[]` categories means "every category".
ROLE_PERMISSIONS: dict[Role, tuple[list[DocumentCategory], Sensitivity]] = {
    Role.VIEWER: (
        [DocumentCategory.POLICY, DocumentCategory.GENERAL],
        Sensitivity.INTERNAL,
    ),
    Role.ANALYST: (
        [DocumentCategory.POLICY, DocumentCategory.RISK, DocumentCategory.GENERAL],
        Sensitivity.CONFIDENTIAL,
    ),
    Role.AUDITOR: (
        [
            DocumentCategory.POLICY,
            DocumentCategory.AUDIT,
            DocumentCategory.RISK,
            DocumentCategory.SECURITY,
            DocumentCategory.GENERAL,
        ],
        Sensitivity.CONFIDENTIAL,
    ),
    Role.ADMIN: ([], Sensitivity.RESTRICTED),
}

WRITE_ROLES = {Role.ADMIN}


def build_user(username: str, role: Role) -> User:
    categories, ceiling = ROLE_PERMISSIONS[role]
    return User(
        username=username,
        role=role,
        allowed_categories=list(categories),
        max_sensitivity=ceiling,
    )


def can_write(user: User) -> bool:
    return user.role in WRITE_ROLES


# --- password hashing --------------------------------------------------


def hash_password(password: str, *, salt: str | None = None) -> str:
    """PBKDF2-HMAC-SHA256. Stored as `pbkdf2$<iterations>$<salt>$<hash>`."""
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS
    )
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2":
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), int(iterations)
    )
    # Constant-time compare so a wrong password cannot be found by timing.
    return hmac.compare_digest(derived.hex(), digest)


# --- user store --------------------------------------------------------


class UserStore:
    """Users persisted as `users.json` next to the index.

    Credentials are never committed: the file is gitignored, and `scripts/seed.py`
    generates passwords at seed time.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._users: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self._users = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._users, indent=2))
        os.chmod(self.path, 0o600)

    def add(self, username: str, password: str, role: Role) -> User:
        self._users[username] = {
            "password_hash": hash_password(password),
            "role": role.value,
        }
        self.save()
        return build_user(username, role)

    def authenticate(self, username: str, password: str) -> User | None:
        record = self._users.get(username)
        if record is None:
            # Hash anyway so a missing user and a wrong password take the
            # same time and cannot be told apart.
            hash_password(password)
            return None
        if not verify_password(password, record["password_hash"]):
            return None
        return build_user(username, Role(record["role"]))

    def get(self, username: str) -> User | None:
        record = self._users.get(username)
        return build_user(username, Role(record["role"])) if record else None

    def __len__(self) -> int:
        return len(self._users)


# --- tokens ------------------------------------------------------------


class InvalidToken(Exception):
    pass


def create_access_token(user: User, settings: Settings) -> tuple[str, int]:
    expires_in = settings.access_token_ttl_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": user.username,
        "role": user.role.value,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_ttl_minutes),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_access_token(token: str, settings: Settings) -> User:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as error:
        raise InvalidToken("Session expired; sign in again.") from error
    except jwt.PyJWTError as error:
        raise InvalidToken("Invalid authentication token.") from error

    username = payload.get("sub")
    role = payload.get("role")
    if not username or role not in Role._value2member_map_:
        raise InvalidToken("Malformed authentication token.")

    # Permissions are rebuilt from the role table, never read from the token,
    # so a tampered or stale token cannot widen access.
    return build_user(username, Role(role))
