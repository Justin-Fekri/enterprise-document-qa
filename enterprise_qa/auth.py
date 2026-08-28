"""Role-based access control for enterprise document collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import jwt

from enterprise_qa.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

ROLES = ("admin", "compliance", "auditor", "employee")

# Document classification → roles that may read it.
CLASSIFICATION_ROLES: dict[str, frozenset[str]] = {
    "public": frozenset(ROLES),
    "internal": frozenset(ROLES),
    "confidential": frozenset({"admin", "compliance", "auditor"}),
    "restricted": frozenset({"admin", "compliance"}),
}


@dataclass(frozen=True)
class User:
    username: str
    full_name: str
    role: str
    password: str  # demo credentials only


# Seeded demo accounts so the UI is usable without a user directory.
USERS: dict[str, User] = {
    "admin": User("admin", "Avery Chen", "admin", "admin123"),
    "compliance": User("compliance", "Jordan Hale", "compliance", "comp123"),
    "auditor": User("auditor", "Sam Okonkwo", "auditor", "audit123"),
    "employee": User("employee", "Riley Patel", "employee", "emp123"),
}


def authenticate(username: str, password: str) -> User | None:
    user = USERS.get(username.strip().lower())
    if user is None or user.password != password:
        return None
    return user


def issue_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "name": user.full_name,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def can_access(role: str, classification: str) -> bool:
    allowed = CLASSIFICATION_ROLES.get(classification, frozenset({"admin"}))
    return role in allowed


def visible_classifications(role: str) -> list[str]:
    return [c for c, roles in CLASSIFICATION_ROLES.items() if role in roles]


def filter_by_role(role: str, items: Iterable, classification_attr: str = "classification"):
    return [item for item in items if can_access(role, getattr(item, classification_attr))]
