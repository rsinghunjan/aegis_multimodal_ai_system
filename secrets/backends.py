Supported backends (priority):
- ENV (default): read from environment variables.
- VAULT: HashiCorp Vault (requires VAULT_ADDR and VAULT_TOKEN).
- AWS_SECRETS_MANAGER: AWS Secrets Manager (requires AWS creds via boto3 chain).

Usage:
    from aegis_multimodal_ai_system.secrets.backends import SecretsManager

    sm = SecretsManager()
    val = sm.get_secret("my/service/api_key", default=None)

Design notes / safety:
- Do NOT log secret values.
- Use env for local dev, Vault/AWS Secrets Manager for staging/prod.
- This module uses lazy imports for optional deps (hvac, boto3).
"""
from typing import Optional
import os
import json
import logging

logger = logging.getLogger(__name__)


class SecretsManager:
    def __init__(self, prefer: Optional[list] = None):
        """
        prefer: optional ordered list of backends to try, e.g. ["vault", "aws", "env"].
        If None, autodetect: vault if VAULT_ADDR present, else aws if AWS env present, else env.
        """
        self.prefer = prefer or self._autodetect_order()
        # Keep clients lazily created
        self._vault_client = None
        self._aws_client = None

    def _autodetect_order(self):
        order = []
        if os.getenv("VAULT_ADDR") and os.getenv("VAULT_TOKEN"):
            order.append("vault")
        # simple heuristic for AWS SDK presence (may be available in EC2/EKS via role)
        if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_ACCESS_KEY_ID"):
            order.append("aws")
        order.append("env")
        return order

    def _get_from_env(self, key: str) -> Optional[str]:
        # simple mapping: allow both exact name and uppercased underscored
        v = os.getenv(key)
        if v is not None:
            return v
        alt = key.upper().replace("-", "_")
        return os.getenv(alt)

    def _init_vault(self):
        if self._vault_client is not None:
            return
        try:
            import hvac  # type: ignore
        except Exception:
            logger.debug("hvac not installed; Vault backend unavailable")
            self._vault_client = None
            return
        addr = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if not addr or not token:
            logger.debug("Vault env not configured (VAULT_ADDR/VAULT_TOKEN)")
            self._vault_client = None
            return
        try:
            client = hvac.Client(url=addr, token=token)
            if not client.is_authenticated():
                logger.warning("Vault client failed to authenticate; check VAULT_TOKEN")
                self._vault_client = None
            else:
                self._vault_client = client
        except Exception:
            logger.exception("Failed to initialize Vault client")
            self._vault_client = None

    def _get_from_vault(self, path: str, key: Optional[str] = None) -> Optional[str]:
        """
        Try reading secret from Vault at `path`. If key provided, return that field, else return
        the full JSON-stringified secret if it is a dict.
        """
        self._init_vault()
        if not self._vault_client:
            return None
        try:
            # try kv v2 first (common)
            if path.startswith("secret/") or path.startswith("kv/") or path.startswith("v2/"):
                # attempt kv v2 read; hvac requires mount path and secret path handling
                # Try generic read first
                try:
                    secret = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
                    data = secret.get("data", {}).get("data", {})
                except Exception:
                    # fallback to generic read
                    secret = self._vault_client.secrets.kv.read_secret_version(path=path)
                    data = secret.get("data", {})
            else:
                # generic secret read
                secret = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
                data = secret.get("data", {}).get("data", {})
        except Exception:
            # last-resort: try logical read
            try:
                secret = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
                data = secret.get("data", {}).get("data", {})
            except Exception:
                logger.debug("Vault read failed for path %s", path)
                return None

        if data is None:
            return None
        if key:
            return data.get(key)
        # if data is dict, return JSON string to keep type consistent
        if isinstance(data, dict):
            return json.dumps(data)
        return str(data)

    def _init_aws(self):
        if self._aws_client is not None:
            return
        try:
            import boto3  # type: ignore
        except Exception:
            logger.debug("boto3 not installed; AWS Secrets Manager backend unavailable")
            self._aws_client = None
            return
        try:
            self._aws_client = boto3.client("secretsmanager")
        except Exception:
            logger.exception("Failed to create AWS Secrets Manager client")
            self._aws_client = None

    def _get_from_aws(self, name: str, key: Optional[str] = None) -> Optional[str]:
        self._init_aws()
        if not self._aws_client:
            return None
        try:
            resp = self._aws_client.get_secret_value(SecretId=name)
            secret_str = resp.get("SecretString")
            if secret_str:
                try:
                    parsed = json.loads(secret_str)
                    if key:
                        return parsed.get(key)
                    return json.dumps(parsed)
                except Exception:
                    # not JSON
                    if key:
                        return None
                    return secret_str
            # binary case
            return None
        except Exception:
            logger.debug("AWS Secrets Manager read failed for %s", name)
            return None

    def get_secret(self, name: str, key: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret named `name`. If secret is structured and you want a field, pass `key`.
        Tries configured backends in order; does not log secret contents.
        """
        for backend in self.prefer:
            if backend == "vault":
                val = self._get_from_vault(name, key=key)
                if val is not None:
                    return val
            elif backend == "aws":
                val = self._get_from_aws(name, key=key)
                if val is not None:
                    return val
            elif backend == "env":
                val = self._get_from_env(name)
                if val is not None:
                    return val
        return default
                        
