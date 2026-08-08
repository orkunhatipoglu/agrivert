"""Auth schemas.

Note on the Firebase choice: with Firebase Auth, the client SDK owns the
credential lifecycle — `signInWithEmailAndPassword` mints the ID token and
`getIdToken(true)` refreshes it, neither of which round-trips through this
API. So of ROUTES.md's five auth routes, only /register and /me have real
server-side work; /login, /refresh and /logout are documented stubs that
explain what the client should call instead. See app/routers/auth.py.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import EmailStr, Field

from app.schemas.common import CamelModel


class RegisterRequest(CamelModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=200)


class UserProfile(CamelModel):
    uid: str
    email: str | None = None
    display_name: str | None = None
    email_verified: bool = False
    disabled: bool = False
    is_admin: bool = False
    created_at: datetime | None = None


class LoginRequest(CamelModel):
    email: EmailStr
    password: str


class RefreshRequest(CamelModel):
    refresh_token: str


class TokenPair(CamelModel):
    access_token: str
    refresh_token: str
    expires_in: int
