
#!/usr/bin/env bash
set -euo pipefail
# Quick local scanner using detect-secrets or truffleHog if installed

if command -v detect-secrets >/dev/null 2>&1; then
  echo "Running detect-secrets scan..."
  detect-secrets scan
elif command -v trufflehog >/dev/null 2>&1; then
  echo "Running trufflehog scan..."
  trufflehog filesystem --no-update --rules . || true
else
  echo "Install detect-secrets (pip install detect-secrets) or trufflehog to run local secret scans."
  exit 2
fi
