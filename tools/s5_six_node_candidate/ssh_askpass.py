#!/usr/bin/env python3
"""Return the process-only SSH secret to OpenSSH; never persist or log it."""
import os

if not os.environ.get("S5_SSH_PASSWORD"):
    raise SystemExit("S5_SSH_PASSWORD is not set; use a public key or explicit runtime authorization")
print(os.environ["S5_SSH_PASSWORD"])
