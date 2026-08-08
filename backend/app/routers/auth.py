"""Auth routes.

STUB STATUS: /register and /me are implemented (the admin SDK genuinely owns
that work). /login, /refresh and /logout return 501 with a pointer to the
client-side call, because with Firebase Auth the client SDK — not this API —
mints and refreshes ID tokens. Proxying passwords through the backend just to
match the route table would mean handling raw credentials for no benefit.

If you later move off Firebase, these three become real.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from firebase_admin import auth as fb_auth

from app.dependencies import CurrentUser, get_current_user, not_implemented
from app.firebase import get_app
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserProfile,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Create a farmer account",
)
async def register(body: RegisterRequest) -> UserProfile:
    """Create the Firebase Auth user. The client then signs in to get tokens."""
    try:
        user = fb_auth.create_user(
            email=body.email,
            password=body.password,
            display_name=body.display_name,
            app=get_app(),
        )
    except fb_auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an account with that email already exists",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return UserProfile(
        uid=user.uid,
        email=user.email,
        display_name=user.display_name,
        email_verified=user.email_verified,
        disabled=user.disabled,
    )


@router.get("/me", response_model=UserProfile, summary="Current user profile")
async def me(user: CurrentUser = Depends(get_current_user)) -> UserProfile:
    record = fb_auth.get_user(user.uid, app=get_app())
    return UserProfile(
        uid=record.uid,
        email=record.email,
        display_name=record.display_name,
        email_verified=record.email_verified,
        disabled=record.disabled,
        is_admin=user.is_admin,
    )


@router.post("/login", response_model=TokenPair, summary="Authenticate (see note)")
async def login(body: LoginRequest) -> TokenPair:
    not_implemented(
        "POST /auth/login",
        "With Firebase Auth the client SDK signs in directly "
        "(signInWithEmailAndPassword) and receives the ID + refresh tokens. "
        "Send the resulting ID token as 'Authorization: Bearer <token>'.",
    )


@router.post("/refresh", response_model=TokenPair, summary="Refresh an access token")
async def refresh(body: RefreshRequest) -> TokenPair:
    not_implemented(
        "POST /auth/refresh",
        "Handled by the Firebase client SDK (getIdToken(true)) or the "
        "securetoken.googleapis.com endpoint.",
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Invalidate tokens")
async def logout(user: CurrentUser = Depends(get_current_user)) -> None:
    not_implemented(
        "POST /auth/logout",
        "Server-side revocation is auth.revoke_refresh_tokens(uid); the "
        "get_current_user dependency already passes check_revoked=True so "
        "revoked tokens are rejected. Wire this up when you decide whether "
        "logout should be global or per-device.",
    )
