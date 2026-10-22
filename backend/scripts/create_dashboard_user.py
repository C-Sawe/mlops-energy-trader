#!/usr/bin/env python3
"""Create a dashboard email-and-password account, change its password, or
delete it (src/serving/auth.py).

Usage:
    python scripts/create_dashboard_user.py --email you@gmail.com
    python scripts/create_dashboard_user.py --email you@gmail.com --delete

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
    parser.add_argument("--delete", action="store_true", help="remove the account instead")
    args = parser.parse_args(argv)
    email = args.email.strip().lower()

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

    password = getpass.getpass(f"New password for {email}: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"password must be at least {MIN_PASSWORD_LENGTH} characters", file=sys.stderr)
        return 1
    if getpass.getpass("Repeat it: ") != password:
        print("passwords differ", file=sys.stderr)
        return 1

    created = repo.set_dashboard_password(email, hash_password(password))
    print(f"{'created' if created else 'password changed for'} {email}")
    if email not in AUTH.allowed_emails:
        print(f"warning: {email} is not in ALLOWED_EMAILS here, so it can't sign in yet")
    if not AUTH.password_login:
        print("warning: PASSWORD_LOGIN is not true here, so password sign-in is off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
