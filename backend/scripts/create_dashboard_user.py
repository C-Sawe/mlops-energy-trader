#!/usr/bin/env python3
"""Create a dashboard email-and-password account, change its password, or
delete it (src/serving/auth.py).

Usage:
    python scripts/create_dashboard_user.py --email you@gmail.com
    python scripts/create_dashboard_user.py --email you@gmail.com --delete

For the deployed site without SSH (docs/DEPLOYMENT.md, "Sign-in"): hash the
password on your own machine and store only the hash as a GitHub secret; the
deploy workflow then creates the account with --hash-stdin.
    python scripts/create_dashboard_user.py --email you@gmail.com --hash-only \
        | gh secret set DASHBOARD_USER_PASSWORD_HASH

The password is typed at a hidden prompt, twice. It is never accepted as an
argument, where it would land in shell history and the process list
(NFR-10). Running it again for an existing email replaces the password.

On the deployed VM:
    docker compose exec backend python scripts/create_dashboard_user.py --email you@gmail.com

An account only signs in if its email is also in ALLOWED_EMAILS and
PASSWORD_LOGIN=true; the script warns about either but doesn't refuse, so
accounts can be prepared before sign-in is switched on.

Exit codes:
    0  success
    1  invalid input (passwords differ, too short, no such account)
    2  database error
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AUTH  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402
from src.serving.auth import MIN_PASSWORD_LENGTH, hash_password  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--delete", action="store_true", help="remove the account instead")
    mode.add_argument("--hash-only", action="store_true",
                      help="prompt for the password and print only its hash; no database is touched")
    mode.add_argument("--hash-stdin", action="store_true",
                      help="store a hash made with --hash-only, read from stdin (used by the deploy)")
    args = parser.parse_args(argv)
    email = args.email.strip().lower()

    if args.hash_only:
        password = _prompt_password(email)
        if password is None:
            return 1
        print(hash_password(password))  # stdout carries the hash only, so it can be piped
        return 0

    try:
        repo = MarketRepository()
        repo.create_schema()
    except Exception as exc:  # noqa: BLE001 - reported, not raised, for a clean CLI exit
        print(f"database error: {exc}", file=sys.stderr)
        return 2

    if args.delete:
        if not repo.delete_dashboard_user(email):
            print(f"no account for {email}", file=sys.stderr)
            return 1
        print(f"deleted {email}")
        return 0

    if args.hash_stdin:
        password_hash = sys.stdin.read().strip()
        if not _looks_like_hash(password_hash):
            print("stdin is not a password hash from --hash-only", file=sys.stderr)
            return 1
    else:
        password = _prompt_password(email)
        if password is None:
            return 1
        password_hash = hash_password(password)

    created = repo.set_dashboard_password(email, password_hash)
    print(f"{'created' if created else 'password changed for'} {email}")
    if email not in AUTH.allowed_emails:
        print(f"warning: {email} is not in ALLOWED_EMAILS here, so it can't sign in yet")
    if not AUTH.password_login:
        print("warning: PASSWORD_LOGIN is not true here, so password sign-in is off")
    return 0


def _prompt_password(email: str) -> str | None:
    """Hidden prompt, twice. None (after saying why) if it is too short or
    the two entries differ."""
    password = getpass.getpass(f"New password for {email}: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"password must be at least {MIN_PASSWORD_LENGTH} characters", file=sys.stderr)
        return None
    if getpass.getpass("Repeat it: ") != password:
        print("passwords differ", file=sys.stderr)
        return None
    return password


def _looks_like_hash(value: str) -> bool:
    """The `hash_password` format: scrypt$N$r$p$salt$key."""
    parts = value.split("$")
    return (len(parts) == 6 and parts[0] == "scrypt" and all(p.isdigit() for p in parts[1:4])
            and all(parts[4:]))


if __name__ == "__main__":
    sys.exit(main())
