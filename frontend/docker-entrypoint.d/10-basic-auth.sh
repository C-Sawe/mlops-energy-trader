#!/bin/sh
# Writes the basic-auth password file nginx.conf.template points at.
# Refuses to start without credentials rather than serving an open dashboard
# (CLAUDE.md §11: prefer explicit failure to silent degradation).
set -eu

if [ -z "${BASIC_AUTH_USER:-}" ] || [ -z "${BASIC_AUTH_PASSWORD:-}" ]; then
    echo "10-basic-auth.sh: BASIC_AUTH_USER and BASIC_AUTH_PASSWORD must be set" >&2
    exit 1
fi

# -i reads the password from stdin so it never appears in a process listing.
printf '%s' "$BASIC_AUTH_PASSWORD" | htpasswd -ciB /etc/nginx/.htpasswd "$BASIC_AUTH_USER"
