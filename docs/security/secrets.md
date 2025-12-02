# Secrets and Supply-chain Guidance for Aegis

This document explains recommended practices and how to configure secrets and supply-chain protections for the Aegis repo.

1) Local / dev secrets (do this)
- Use environment variables for local development. Store them in a local .env file only if you never commit it.
- Add .env to .gitignore.
- Use the SecretsManager abstraction (aegis_multimodal_ai_system.secrets.backends.SecretsManager)
  to fetch secrets; tests and local dev can still use env vars.

2) CI secrets (GitHub Actions)
- Add required values in the repository "Settings → Secrets and variables → Actions".
  - e.g., VAULT_ADDR, VAULT_TOKEN (if you use Vault), AWS credentials (if using AWS), COSIGN_PUBKEY for image verify.
- In Actions workflows, reference them via `${{ secrets.MY_SECRET }}`.
- Do not print secret values in logs.

3) Production secrets (recommended)
- Use a secrets manager:
  - HashiCorp Vault: recommended for enterprise control. Use auto-unseal and short-lived tokens.
  - AWS Secrets Manager / Parameter Store: good integration with IAM roles.
- Configure the runtime to use the corresponding backend (VAULT_ADDR/VAULT_TOKEN, or IAM role for AWS).
- Avoid long-lived static credentials.

4) Pre-commit & local scanning
- Install pre-commit and run `pre-commit install`.
- Keep a detect-secrets baseline in the repo (.secrets.baseline). Update only after manual review.

5) Supply-chain protections
- Enable Dependabot for pip (dependabot.yml included).
- Add the Security CI workflow (security-ci.yml) to run pip-audit and bandit on PRs.
- Sign production images/artifacts with cosign and enforce verification in CI/CD.

6) Branch protections & policy
- Enable branch protection on `main`:
  - Require status checks to pass (security-ci + unit tests).
  - Require pull request reviews before merging.
  - Require signed commits if desired.
- Restrict who can push to protected branches.

7) Image signing (sigstore / cosign)
- Generate a keypair with cosign and upload the public key to repository secrets (COSIGN_PUBKEY).
- In CI, verify the images before promoting them.

8) Incident response (quick)
- If a secret is accidentally committed:
  - Immediately rotate that secret (revoke & recreate).
  - Remove the secret from git history (git filter-repo or BFG) and force-push.
  - Run a secret-scan to ensure no other occurrences.
  - Notify security/compliance and follow escalation steps in runbook.

9) Audit & logs
- Ensure audit logs are written to a secure backend (S3 with restricted IAM policy or DB).
- Regularly prune and backup (see audit/prune.py).

10) Helpful commands
- Run local scan: `./scripts/local-secret-scan.sh`
- Install pre-commit: `pip install pre-commit && pre-commit install`
- Run security CI locally: run bandit, pip-audit, flake8 manually.

If you want, I can:
- Add sample GitHub Actions secrets documentation with exact key names for your repo.
- Add a small script to rotate API keys (generate, store, notify).
- Add cosign signing and verification steps in your release workflow (requires write access to push changes).
