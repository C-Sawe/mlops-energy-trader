"""Tests for dashboard sign-in (src/serving/auth.py): FR-21, FR-22, NFR-12.

No network (CLAUDE.md §11): the provider round-trip is replaced by
monkeypatching `auth.fetch_identity`, which is the one function that talks
to Google, Microsoft or GitHub. What's tested is everything around it —
the allowlist, the session, the guard on the API, and which email each
provider's response is trusted for.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from src.dataops.repository import MarketRepository
from src.serving import auth
from src.serving.api import app
from tests.test_api import _seed_db

ALLOWED = "caleb@example.com"
PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def _reset_login_throttle():
    auth._failed_logins.clear()
    yield
    auth._failed_logins.clear()


def _enable_sign_in(monkeypatch, tmp_path, *, providers=("google",), allowed=ALLOWED):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    for provider in providers:
        monkeypatch.setenv(f"{provider.upper()}_CLIENT_ID", f"{provider}-id")
        monkeypatch.setenv(f"{provider.upper()}_CLIENT_SECRET", f"{provider}-secret")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("ALLOWED_EMAILS", allowed)
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:5173")
    monkeypatch.setenv("PUBLIC_API_URL", "http://localhost:8000")


def _signs_in_as(monkeypatch, email: str | None):
    async def fake(provider, request):
        return {"email": email, "name": "Caleb", "provider": provider} if email else None

    monkeypatch.setattr(auth, "fetch_identity", fake)


# --------------------------------------------------------------- config
def test_sign_in_is_off_and_the_api_open_without_providers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    with TestClient(app) as client:
        session = client.get("/auth/session").json()
        assert session == {"auth_enabled": False, "providers": [], "password_login": False, "user": None}
        assert client.get("/ct-status").status_code == 200


def test_only_fully_configured_providers_are_offered(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path, providers=("google",))
    monkeypatch.setenv("GITHUB_CLIENT_ID", "id-without-a-secret")

    with TestClient(app) as client:
        providers = client.get("/auth/session").json()["providers"]
        assert providers == [{"id": "google", "label": "Google"}]
        assert client.get("/auth/login/github", follow_redirects=False).status_code == 404


@pytest.mark.parametrize("unset", ["SESSION_SECRET", "ALLOWED_EMAILS"])
def test_startup_refuses_sign_in_without_secret_or_allowlist(tmp_path, monkeypatch, unset):
    _enable_sign_in(monkeypatch, tmp_path)
    monkeypatch.delenv(unset)

    with pytest.raises(RuntimeError, match=unset):
        with TestClient(app):
            pass


# --------------------------------------------------------------- flow
def test_login_redirects_to_the_provider_with_the_public_callback(tmp_path, monkeypatch):
    """GitHub has no discovery document, so this builds the real redirect
    without any network call."""
    _enable_sign_in(monkeypatch, tmp_path, providers=("github",))

    with TestClient(app) as client:
        resp = client.get("/auth/login/github", follow_redirects=False)
        location = urlparse(resp.headers["location"])
        query = parse_qs(location.query)
        assert f"{location.netloc}{location.path}" == "github.com/login/oauth/authorize"
        assert query["redirect_uri"] == ["http://localhost:8000/auth/callback/github"]
        assert query["client_id"] == ["github-id"]
        assert query["state"]  # Authlib checks it on the way back


def test_allowlisted_sign_in_opens_the_api_until_sign_out(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)
    _signs_in_as(monkeypatch, ALLOWED)

    with TestClient(app) as client:
        assert client.get("/ct-status").status_code == 401

        resp = client.get("/auth/callback/google", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "http://localhost:5173/"

        user = client.get("/auth/session").json()["user"]
        assert user["email"] == ALLOWED
        assert client.get("/ct-status").status_code == 200

        assert client.post("/auth/logout").status_code == 204
        assert client.get("/auth/session").json()["user"] is None
        assert client.get("/ct-status").status_code == 401


def test_email_not_on_the_allowlist_is_refused(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)
    _signs_in_as(monkeypatch, "stranger@example.com")

    with TestClient(app) as client:
        resp = client.get("/auth/callback/google", follow_redirects=False)
        assert resp.headers["location"] == "http://localhost:5173/?auth_error=not_allowed"
        assert client.get("/auth/session").json()["user"] is None
        assert client.get("/ct-status").status_code == 401


def test_account_without_a_verified_email_is_refused(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)
    _signs_in_as(monkeypatch, None)

    with TestClient(app) as client:
        resp = client.get("/auth/callback/google", follow_redirects=False)
        assert resp.headers["location"].endswith("?auth_error=unverified")
        assert client.get("/ct-status").status_code == 401


def test_a_failed_provider_exchange_returns_to_the_login_page(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)

    async def broken(provider, request):
        raise RuntimeError("mismatching_state")

    monkeypatch.setattr(auth, "fetch_identity", broken)

    with TestClient(app) as client:
        resp = client.get("/auth/callback/google", follow_redirects=False)
        assert resp.headers["location"].endswith("?auth_error=failed")


def test_removing_an_email_from_the_allowlist_revokes_its_session(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path, allowed=f"{ALLOWED},supervisor@example.com")
    _signs_in_as(monkeypatch, ALLOWED)

    with TestClient(app) as client:
        client.get("/auth/callback/google", follow_redirects=False)
        assert client.get("/ct-status").status_code == 200

        monkeypatch.setenv("ALLOWED_EMAILS", "supervisor@example.com")
        assert client.get("/ct-status").status_code == 401


def test_bearer_token_still_works_with_sign_in_enabled(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)
    monkeypatch.setenv("API_BEARER_TOKEN", "secret123")

    with TestClient(app) as client:
        assert client.get("/ct-status").status_code == 401
        resp = client.get("/ct-status", headers={"Authorization": "Bearer secret123"})
        assert resp.status_code == 200


def test_health_and_session_stay_public_with_sign_in_enabled(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/auth/session").json()["auth_enabled"] is True


# --------------------------------------------------------------- trusted email
def test_google_email_requires_email_verified():
    assert auth.google_email({"email": "A@x.com", "email_verified": True}) == "a@x.com"
    assert auth.google_email({"email": "a@x.com", "email_verified": False}) is None
    assert auth.google_email({"email": "a@x.com"}) is None


def test_microsoft_trusts_email_only_for_personal_accounts():
    personal = {"tid": auth.CONSUMER_TENANT_ID, "email": "me@outlook.com", "preferred_username": "x"}
    assert auth.microsoft_email(personal) == "me@outlook.com"

    # A work/school tenant's admin can set `email` to anything (nOAuth);
    # the UPN's domain is one the tenant had to verify.
    work = {"tid": "some-tenant", "email": "victim@example.com", "preferred_username": "Me@Strathmore.edu"}
    assert auth.microsoft_email(work) == "me@strathmore.edu"


def test_github_uses_the_primary_verified_address_only():
    emails = [
        {"email": "public@x.com", "primary": False, "verified": True},
        {"email": "unverified@x.com", "primary": True, "verified": False},
    ]
    assert auth.github_email(emails) is None
    emails.append({"email": "Main@x.com", "primary": True, "verified": True})
    assert auth.github_email(emails) == "main@x.com"


def test_microsoft_issuer_must_match_the_tokens_own_tenant():
    validate = auth._microsoft_claims_options()["iss"]["validate"]
    claims = {"tid": "tenant-a"}
    assert validate(claims, "https://login.microsoftonline.com/tenant-a/v2.0")
    assert not validate(claims, "https://login.microsoftonline.com/tenant-b/v2.0")


# --------------------------------------------------------------- email + password
def _enable_password_login(monkeypatch, tmp_path, *, allowed=ALLOWED, account=ALLOWED):
    _enable_sign_in(monkeypatch, tmp_path, providers=())
    monkeypatch.setenv("PASSWORD_LOGIN", "true")
    monkeypatch.setenv("ALLOWED_EMAILS", allowed)
    if account:
        MarketRepository().set_dashboard_password(account, auth.hash_password(PASSWORD))


def _login(client, email=ALLOWED, password=PASSWORD):
    return client.post("/auth/password/login", json={"email": email, "password": password})


def test_password_hash_is_salted_and_verifies_only_the_right_password():
    first, second = auth.hash_password(PASSWORD), auth.hash_password(PASSWORD)
    assert first != second and PASSWORD not in first
    assert auth.verify_password(PASSWORD, first)
    assert not auth.verify_password("wrong password", first)
    assert not auth.verify_password(PASSWORD, "not-a-hash")


def test_password_login_alone_turns_sign_in_on(tmp_path, monkeypatch):
    _enable_password_login(monkeypatch, tmp_path)

    with TestClient(app) as client:
        session = client.get("/auth/session").json()
        assert session["auth_enabled"] is True
        assert session["password_login"] is True
        assert session["providers"] == []
        assert client.get("/ct-status").status_code == 401


def test_correct_password_signs_in_until_sign_out(tmp_path, monkeypatch):
    _enable_password_login(monkeypatch, tmp_path)

    with TestClient(app) as client:
        resp = _login(client, email="  Caleb@Example.com ")
        assert resp.status_code == 200
        assert resp.json() == {"email": ALLOWED, "name": None, "provider": "password"}
        assert client.get("/ct-status").status_code == 200

        client.post("/auth/logout")
        assert client.get("/ct-status").status_code == 401


@pytest.mark.parametrize(
    "email, password",
    [(ALLOWED, "wrong password"), ("nobody@example.com", PASSWORD)],
    ids=["wrong-password", "unknown-email"],
)
def test_failed_password_logins_look_the_same(tmp_path, monkeypatch, email, password):
    """No way to tell an unknown email from a wrong password."""
    _enable_password_login(monkeypatch, tmp_path, allowed=f"{ALLOWED},nobody@example.com")

    with TestClient(app) as client:
        resp = _login(client, email=email, password=password)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Incorrect email or password."
        assert client.get("/ct-status").status_code == 401


def test_an_account_not_on_the_allowlist_cannot_sign_in(tmp_path, monkeypatch):
    _enable_password_login(monkeypatch, tmp_path, allowed="someone-else@example.com")

    with TestClient(app) as client:
        assert _login(client).status_code == 401


def test_repeated_failures_are_throttled_even_for_the_right_password(tmp_path, monkeypatch):
    _enable_password_login(monkeypatch, tmp_path)

    with TestClient(app) as client:
        for _ in range(auth.MAX_FAILED_LOGINS):
            assert _login(client, password="guess").status_code == 401
        assert _login(client).status_code == 429

        # The window passing lifts it.
        auth._failed_logins[ALLOWED] = [t - auth.FAILED_LOGIN_WINDOW_SECONDS for t in auth._failed_logins[ALLOWED]]
        assert _login(client).status_code == 200


def test_password_login_is_404_when_not_enabled(tmp_path, monkeypatch):
    _enable_sign_in(monkeypatch, tmp_path)  # Google only

    with TestClient(app) as client:
        assert _login(client).status_code == 404


def test_malformed_password_login_is_rejected_not_crashed(tmp_path, monkeypatch):
    """NFR-08."""
    _enable_password_login(monkeypatch, tmp_path)

    with TestClient(app) as client:
        assert client.post("/auth/password/login", json={"email": ALLOWED}).status_code == 422
        too_long = {"email": ALLOWED, "password": "x" * 5000}
        assert client.post("/auth/password/login", json=too_long).status_code == 422


def test_create_dashboard_user_script(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_dashboard_user.py"
    spec = importlib.util.spec_from_file_location("create_dashboard_user", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("ALLOWED_EMAILS", ALLOWED)
    typed = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr(script.getpass, "getpass", lambda prompt="": next(typed))

    assert script.main(["--email", "Caleb@Example.com"]) == 0
    stored = MarketRepository().get_dashboard_password_hash(ALLOWED)
    assert auth.verify_password(PASSWORD, stored)
    assert "PASSWORD_LOGIN is not true" in capsys.readouterr().out

    typed = iter(["short", "short"])
    assert script.main(["--email", ALLOWED]) == 1
    typed = iter([PASSWORD, "different password!"])
    assert script.main(["--email", ALLOWED]) == 1

    assert script.main(["--email", ALLOWED, "--delete"]) == 0
    assert MarketRepository().get_dashboard_password_hash(ALLOWED) is None


def test_the_deployed_account_is_made_from_a_hash_never_the_password(tmp_path, monkeypatch, capsys):
    """--hash-only prints a hash on the operator's machine (no database);
    --hash-stdin stores it on the server. The password itself never travels."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_dashboard_user.py"
    spec = importlib.util.spec_from_file_location("create_dashboard_user", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'deployed.db'}")
    typed = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr(script.getpass, "getpass", lambda prompt="": next(typed))

    assert script.main(["--email", ALLOWED, "--hash-only"]) == 0
    printed = capsys.readouterr().out.strip()
    assert PASSWORD not in printed and auth.verify_password(PASSWORD, printed)
    assert not (tmp_path / "deployed.db").exists()  # hashing touches no database

    monkeypatch.setattr(script.sys, "stdin", io.StringIO(printed + "\n"))
    assert script.main(["--email", ALLOWED, "--hash-stdin"]) == 0
    assert auth.verify_password(PASSWORD, MarketRepository().get_dashboard_password_hash(ALLOWED))

    monkeypatch.setattr(script.sys, "stdin", io.StringIO(PASSWORD))  # a raw password is refused
    assert script.main(["--email", ALLOWED, "--hash-stdin"]) == 1
