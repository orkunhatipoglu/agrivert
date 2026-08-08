"""Shared FastAPI dependencies.

The auth dependency is fully implemented even though the /auth/* routes are
stubs — the diagnoses pipeline is real, and a real pipeline with fake auth
would let any caller read any farmer's photos.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as fb_auth

from app.firebase import get_app

log = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


class CurrentUser:
    """The authenticated caller, from a verified Firebase ID token."""

    def __init__(self, uid: str, email: str | None, claims: dict):
        self.uid = uid
        self.email = email
        self.claims = claims

    @property
    def is_admin(self) -> bool:
        # Set with: auth.set_custom_user_claims(uid, {"admin": True})
        return bool(self.claims.get("admin", False))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    """Verify the Firebase ID token in the Authorization header."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # check_revoked catches logout/disable between token issue and use.
        decoded = fb_auth.verify_id_token(
            credentials.credentials, app=get_app(), check_revoked=True
        )
    except fb_auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="token has been revoked"
        )
    except fb_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="token has expired"
        )
    except Exception as exc:
        log.warning("token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token"
        )

    return CurrentUser(
        uid=decoded["uid"], email=decoded.get("email"), claims=decoded
    )


async def require_admin(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Gate the /admin routes on a custom `admin` claim."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="admin privileges required"
        )
    return user


def owned_or_404(record: dict | None, user: CurrentUser, what: str = "resource") -> dict:
    """Return the record, or 404 if missing OR not the caller's.

    Deliberately 404 rather than 403 on the ownership failure: a 403 would
    confirm the id exists, letting someone enumerate other farmers' records.
    """
    if record is None or record.get("owner_uid") != user.uid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found"
        )
    return record


def not_implemented(route: str, reason: str = "") -> None:
    """Uniform 501 for the scaffolded-but-unimplemented routes."""
    detail = f"{route} is scaffolded but not implemented yet."
    if reason:
        detail += f" {reason}"
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail
    )
