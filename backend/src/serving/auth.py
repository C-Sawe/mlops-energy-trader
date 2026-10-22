"""Dashboard sign-in (FR-21, FR-22, NFR-12): Google, Microsoft and GitHub
over OAuth 2.0 / OpenID Connect, and email and password, all gated by one
email allowlist (`AuthConfig`, src/config.py).

Requirements added after the proposal's approval, so recorded as a
deviation in Chapter 5 (CLAUDE.md §2, §5). It replaces the deployment's
shared basic-auth password with per-person identities. The bearer token
(`ServingConfig.bearer_token`) stays, for scripts and server-to-server calls.

The session is Starlette's signed cookie (`SessionMiddleware`, installed in
src/serving/api.py). It holds the signed-in identity, never a provider
token. The allowlist is re-checked on every request, so removing an address
revokes access at once rather than when the cookie expires.

The allowlist is only as strong as the email each provider is trusted for:
- Google: `email`, only when `email_verified` is true.
- GitHub: the primary *verified* address from /user/emails. The profile's
  public email is user-editable and unverified.
- Microsoft: a personal account's `email` is verified by Microsoft. A work
  or school tenant's `email` claim is set by that tenant's admin and is not
  verified ("nOAuth"), so those accounts use `preferred_username`, the UPN,
  whose domain the tenant must have verified.

Password accounts are created by an operator (scripts/create_dashboard_user.py),
never by self-sign-up: without sending a verification email, sign-up would
let anyone claim an allowlisted address. Passwords are stored as salted
scrypt hashes (stdlib `hashlib.scrypt`). Repeated failures for one email are
throttled, and every failure gets the same message and roughly the same
time, so a probe can't tell an unknown email from a wrong password.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
from functools import lru_cache

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from src.config import AUTH
from src.serving.schemas import AuthProvider, AuthSessionResponse, AuthUser, PasswordLoginRequest

# The fixed tenant every personal Microsoft account signs in through.
CONSUMER_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"

PROVIDER_LABELS = {"google": "Google", "microsoft": "Microsoft", "github": "GitHub"}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# scrypt cost: 2**15 x r=8 uses 32 MiB and roughly 50-100 ms per check,
# OWASP's recommended minimum. Stored with the hash, so raising it later
# leaves existing hashes verifiable.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**15, 8, 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
MIN_PASSWORD_LENGTH = 12

# Per email: this many failures inside the window, then 429 until it passes.
MAX_FAILED_LOGINS = 10
FAILED_LOGIN_WINDOW_SECONDS = 15 * 60
_failed_logins: dict[str, list[float]] = {}
_failed_logins_lock = threading.Lock()


def hash_password(password: str) -> str:
    """NFR-12."""
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, maxmem=_SCRYPT_MAXMEM
    )
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
            maxmem=_SCRYPT_MAXMEM, dklen=len(expected),
        )
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


@lru_cache(maxsize=1)
def _decoy_hash() -> str:
    """Checked against when the email has no account, so that case costs
    the same time as a wrong password."""
    return hash_password(secrets.token_urlsafe(16))


def _recent_failures(email: str) -> list[float]:
    cutoff = time.monotonic() - FAILED_LOGIN_WINDOW_SECONDS
    recent = [t for t in _failed_logins.get(email, []) if t > cutoff]
    _failed_logins[email] = recent
    return recent


def _normalise(email: str | None) -> str | None:
    return email.strip().lower() if email and email.strip() else None


def google_email(claims: dict) -> str | None:
    return _normalise(claims.get("email")) if claims.get("email_verified") is True else None


def microsoft_email(claims: dict) -> str | None:
    if claims.get("tid") == CONSUMER_TENANT_ID:
        return _normalise(claims.get("email"))
    return _normalise(claims.get("preferred_username"))


def github_email(emails: list[dict]) -> str | None:
    for entry in emails:
        if entry.get("primary") and entry.get("verified"):
            return _normalise(entry.get("email"))
    return None


@lru_cache(maxsize=None)
def _oauth_client(provider: str, client_id: str, client_secret: str, tenant: str):
    """Cached per credential set, so provider metadata and signing keys are
    fetched once per process rather than on every sign-in."""
    oauth = OAuth()
    if provider == "google":
        oauth.register(
            "google",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    elif provider == "microsoft":
        oauth.register(
            "microsoft",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=(
                f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email profile"},
        )
    else:
        oauth.register(
            "github",
            client_id=client_id,
            client_secret=client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            access_token_url="https://github.com/login/oauth/access_token",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
        )
    return oauth.create_client(provider)


def _client(provider: str):
    client_id, client_secret = AUTH.credentials(provider)
    return _oauth_client(provider, client_id, client_secret, AUTH.microsoft_tenant)


def _microsoft_claims_options() -> dict:
    """The multi-tenant metadata advertises the issuer as the template
    ".../{tenantid}/v2.0", which never equals a real token's `iss`; check
    it against the token's own `tid` instead. Built fresh per call: Authlib
    pops the `validate` hook out of the dict it is given."""
    return {
        "iss": {
            "essential": True,
            "validate": lambda claims, iss: iss
            == f"https://login.microsoftonline.com/{claims.get('tid')}/v2.0",
        }
    }


async def fetch_identity(provider: str, request: Request) -> dict | None:
    """Completes the code exchange (Authlib checks `state`, and for OIDC the
    ID token's signature, audience, nonce and expiry) and returns the
    identity, or None when the account has no email this module trusts."""
    client = _client(provider)
    if provider == "github":
        token = await client.authorize_access_token(request)
        profile = await client.get("user", token=token)
        profile.raise_for_status()
        emails = await client.get("user/emails", token=token)
        emails.raise_for_status()
        email = github_email(emails.json())
        name = profile.json().get("name") or profile.json().get("login")
    elif provider == "microsoft":
        token = await client.authorize_access_token(
            request, claims_options=_microsoft_claims_options()
        )
        email, name = microsoft_email(token["userinfo"]), token["userinfo"].get("name")
    else:
        token = await client.authorize_access_token(request)
        email, name = google_email(token["userinfo"]), token["userinfo"].get("name")
    return {"email": email, "name": name, "provider": provider} if email else None


def current_user(request: Request) -> dict | None:
    """FR-22: the allowlist is checked on every request, not only at sign-in."""
    user = request.session.get("user")
    if not user or user.get("email") not in AUTH.allowed_emails:
        return None
    return user


def _back_to_dashboard(error: str | None = None) -> RedirectResponse:
    query = f"?auth_error={error}" if error else ""
    return RedirectResponse(f"{AUTH.frontend_url}/{query}", status_code=303)


def _require_enabled(provider: str) -> None:
    if provider not in AUTH.enabled_providers:
        raise HTTPException(status_code=404, detail="unknown or disabled sign-in provider")


@router.get("/session", response_model=AuthSessionResponse)
def session_info(request: Request) -> AuthSessionResponse:
    """Public: the dashboard calls this first to decide between the login
    page and the dashboard."""
    user = current_user(request)
    return AuthSessionResponse(
        auth_enabled=AUTH.enabled,
        providers=[AuthProvider(id=p, label=PROVIDER_LABELS[p]) for p in AUTH.enabled_providers],
        password_login=AUTH.password_login,
        user=AuthUser(**user) if user else None,
    )


@router.get("/login/{provider}")
async def login(provider: str, request: Request):
    _require_enabled(provider)
    redirect_uri = f"{AUTH.public_api_url}/auth/callback/{provider}"
    return await _client(provider).authorize_redirect(request, redirect_uri)


@router.get("/callback/{provider}")
async def callback(provider: str, request: Request) -> RedirectResponse:
    _require_enabled(provider)
    try:
        identity = await fetch_identity(provider, request)
    except Exception:  # noqa: BLE001 - any failure here means "not signed in", shown on the login page
        logger.exception("sign-in via %s failed", provider)
        return _back_to_dashboard("failed")
    if identity is None:
        logger.warning("sign-in via %s refused: no verified email on the account", provider)
        return _back_to_dashboard("unverified")
    if identity["email"] not in AUTH.allowed_emails:
        logger.warning("sign-in via %s refused: %s is not in ALLOWED_EMAILS", provider, identity["email"])
        return _back_to_dashboard("not_allowed")
    request.session.clear()
    request.session["user"] = identity
    logger.info("signed in via %s: %s", provider, identity["email"])
    return _back_to_dashboard()


@router.post("/password/login", response_model=AuthUser)
def password_login(body: PasswordLoginRequest, request: Request) -> AuthUser:
    """A plain def: FastAPI runs it in a worker thread, so the deliberately
    slow scrypt check never blocks the event loop serving /predict."""
    if not AUTH.password_login:
        raise HTTPException(status_code=404, detail="password sign-in is not enabled")
    email = _normalise(body.email) or ""
    with _failed_logins_lock:
        if len(_recent_failures(email)) >= MAX_FAILED_LOGINS:
            raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")

    stored = request.app.state.repo.get_dashboard_password_hash(email)
    valid = verify_password(body.password, stored or _decoy_hash())
    if not (stored and valid and email in AUTH.allowed_emails):
        with _failed_logins_lock:
            _recent_failures(email).append(time.monotonic())
        logger.warning("password sign-in refused for %s", email)
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    with _failed_logins_lock:
        _failed_logins.pop(email, None)
    identity = {"email": email, "name": None, "provider": "password"}
    request.session.clear()
    request.session["user"] = identity
    logger.info("signed in with password: %s", email)
    return AuthUser(**identity)


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    request.session.clear()
    return Response(status_code=204)
