"""The security-critical tests.

A passage a user may not read must never be retrieved - because anything
retrieved is sent to the model and can end up quoted in an answer.
"""

import pytest

from app.retrieval.retriever import user_can_read
from app.schemas import Role, Sensitivity
from app.security import (
    ROLE_PERMISSIONS,
    build_user,
    can_write,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_viewer_cannot_retrieve_restricted_content(retriever):
    """The central guarantee of the access model."""
    viewer = build_user("viewer", Role.VIEWER)
    results = retriever.retrieve("break-glass escrow credentials", user=viewer, top_k=10)
    assert all(item.chunk.filename != "access.pdf" for item in results)
    assert all(item.chunk.sensitivity != Sensitivity.RESTRICTED for item in results)


def test_admin_can_retrieve_restricted_content(retriever):
    admin = build_user("admin", Role.ADMIN)
    results = retriever.retrieve("break-glass escrow credentials", user=admin, top_k=10)
    assert any(item.chunk.filename == "access.pdf" for item in results)


def test_viewer_cannot_see_confidential_risk_reports(retriever):
    viewer = build_user("viewer", Role.VIEWER)
    results = retriever.retrieve("risk appetite threshold", user=viewer, top_k=10)
    assert all(item.chunk.filename != "risk.docx" for item in results)


def test_analyst_sees_risk_but_not_restricted_security(retriever):
    analyst = build_user("analyst", Role.ANALYST)
    filenames = {
        item.chunk.filename
        for query in ("risk appetite", "break-glass escrow")
        for item in retriever.retrieve(query, user=analyst, top_k=10)
    }
    assert "risk.docx" in filenames
    assert "access.pdf" not in filenames


def test_anonymous_access_sees_only_public_documents(retriever):
    assert retriever.retrieve("password rotation", user=None, top_k=10) == []


@pytest.mark.parametrize("role", list(Role))
def test_every_role_has_a_permission_entry(role):
    assert role in ROLE_PERMISSIONS
    user = build_user("someone", role)
    assert user.max_sensitivity in Sensitivity


def test_only_admins_may_write():
    assert can_write(build_user("a", Role.ADMIN))
    assert not can_write(build_user("b", Role.AUDITOR))


def test_user_can_read_respects_the_sensitivity_ceiling(seeded_store):
    viewer = build_user("viewer", Role.VIEWER)
    restricted = next(
        chunk for chunk in seeded_store.chunks if chunk.sensitivity == Sensitivity.RESTRICTED
    )
    assert not user_can_read(restricted, viewer)


# --- credentials and tokens -------------------------------------------


def test_password_hashing_round_trip():
    stored = hash_password("correct horse battery staple")
    assert stored.startswith("pbkdf2$")
    assert "correct horse battery staple" not in stored
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong password", stored)


def test_malformed_hash_is_rejected():
    assert not verify_password("anything", "not-a-real-hash")


def test_token_round_trip(settings):
    user = build_user("auditor", Role.AUDITOR)
    token, expires_in = create_access_token(user, settings)
    assert expires_in > 0
    decoded = decode_access_token(token, settings)
    assert decoded.username == "auditor"
    assert decoded.role is Role.AUDITOR


def test_permissions_are_rebuilt_from_the_role_not_read_from_the_token(settings):
    """A tampered token cannot widen access - only the role is trusted."""
    import jwt

    forged = jwt.encode(
        {
            "sub": "viewer",
            "role": Role.VIEWER.value,
            "max_sensitivity": "restricted",
            "allowed_categories": [],
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    user = decode_access_token(forged, settings)
    assert user.max_sensitivity is Sensitivity.INTERNAL


def test_token_signed_with_another_secret_is_rejected(settings):
    from app.security import InvalidToken

    other = settings.model_copy(update={"jwt_secret": "different-secret"})
    token, _ = create_access_token(build_user("admin", Role.ADMIN), other)
    with pytest.raises(InvalidToken):
        decode_access_token(token, settings)


def test_user_store_authenticates(settings):
    from app.security import UserStore

    settings.ensure_dirs()
    users = UserStore(settings.index_dir / "users.json")
    users.add("auditor", "s3cret-password", Role.AUDITOR)

    assert users.authenticate("auditor", "s3cret-password") is not None
    assert users.authenticate("auditor", "wrong") is None
    assert users.authenticate("ghost", "anything") is None
