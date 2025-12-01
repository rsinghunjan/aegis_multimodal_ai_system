"""
Configurable audit logger.

Supported backends:
- file (default) : newline-delimited JSON file at logs/safety_audit.log (safe default)
- s3             : writes each event as a separate object under s3://{AUDIT_S3_BUCKET}/{AUDIT_S3_PREFIX}/
- sqlite         : stores events in a local sqlite DB logs/audit_events.db

Configuration (environment variables):
- AUDIT_BACKEND : "file" | "s3" | "sqlite"   (default: "file")
- AUDIT_RETENTION_DAYS : integer days to keep events for prune jobs (default: 90)

S3-specific:
- AUDIT_S3_BUCKET : required when AUDIT_BACKEND=s3
- AUDIT_S3_PREFIX : optional prefix in bucket (default: "aegis/audit")
- AWS credentials should be provided via the normal boto3 chain (env, profile, IAM role).

SQLite-specific:
- AUDIT_DB_PATH : optional path (default: logs/audit_events.db)

Usage:
- import audit_event from this module and call audit_event(dict)
- Use prune_old(backend_specific_args) to delete old events per retention policy
"""
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

AUDIT_BACKEND = os.getenv("AUDIT_BACKEND", "file").lower()
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))

# Backend modules imported lazily to keep deps optional
FILE_LOG_DIR = Path(os.getenv("AUDIT_FILE_DIR", "logs"))
FILE_LOG_DIR.mkdir(parents=True, exist_ok=True)
FILE_LOG_FILE = FILE_LOG_DIR / "safety_audit.log"

# S3 config
AUDIT_S3_BUCKET = os.getenv("AUDIT_S3_BUCKET", "")
AUDIT_S3_PREFIX = os.getenv("AUDIT_S3_PREFIX", "aegis/audit")

# SQLite config
AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", "logs/audit_events.db"))
AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class AuditBackend:
    def write(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def prune_older_than(self, seconds: int) -> None:
        """
        Remove events older than `seconds`. Optional for backends that don't support pruning.
        """
        raise NotImplementedError()


class FileBackend(AuditBackend):
    def __init__(self, path: Path):
        self.path = path
        # Ensure file exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def write(self, event: Dict[str, Any]) -> None:
        try:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            # append newline-delimited JSON
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + os.linesep)
        except Exception:
            logger.exception("FileBackend: failed to write audit event")

    def prune_older_than(self, seconds: int) -> None:
        """
        Prune file by rewriting with only recent events. This rewrites the file atomically.
        """
        try:
            cutoff = time.time() - seconds
            temp_path = self.path.with_suffix(".tmp")
            with open(self.path, "r", encoding="utf-8") as src, open(temp_path, "w", encoding="utf-8") as dst:
                for line in src:
                    try:
                        obj = json.loads(line)
                        ts = float(obj.get("timestamp", 0))
                        if ts >= cutoff:
                            dst.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + os.linesep)
                    except Exception:
                        # Skip corrupt lines but log
                        logger.warning("FileBackend: skipping corrupt audit line")
                        continue
            os.replace(temp_path, self.path)
        except FileNotFoundError:
            return
        except Exception:
            logger.exception("FileBackend: prune failed")


class SQLiteBackend(AuditBackend):
    def __init__(self, db_path: Path):
        import sqlite3

        self.db_path = db_path
        self._conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        # enable WAL for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                event_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def write(self, event: Dict[str, Any]) -> None:
        try:
            import sqlite3

            eid = event.get("request_id") or str(uuid.uuid4())
            ts = float(event.get("timestamp", time.time()))
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            # Insert (replace) to avoid PK errors
            self._conn.execute(
                "INSERT OR REPLACE INTO audit_events (id, timestamp, event_json) VALUES (?, ?, ?)",
                (eid, ts, payload),
            )
            self._conn.commit()
        except Exception:
            logger.exception("SQLiteBackend: failed to write audit event")

    def prune_older_than(self, seconds: int) -> None:
        try:
            cutoff = time.time() - seconds
            self._conn.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff,))
            self._conn.commit()
        except Exception:
            logger.exception("SQLiteBackend: prune failed")


class S3Backend(AuditBackend):
    def __init__(self, bucket: str, prefix: str):
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except Exception:
            raise RuntimeError("S3Backend requires boto3; install boto3 and configure credentials")

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        self.s3 = boto3.client("s3")

        # Basic sanity: verify bucket exists (do not create bucket to avoid permission surprises)
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception as e:
            # allow instantiation but log warning; writing will likely fail
            logger.warning("S3Backend: head_bucket failed; writes may fail: %s", str(e))

    def _object_key(self, eid: str, ts: float) -> str:
        # e.g., aegis/audit/2025-12-01/16789123-uuid.json
        t = time.gmtime(ts)
        date_path = time.strftime("%Y/%m/%d", t)
        filename = f"{int(ts)}-{eid}.json"
        return f"{self.prefix}/{date_path}/{filename}"

    def write(self, event: Dict[str, Any]) -> None:
        try:
            import botocore

            eid = event.get("request_id") or str(uuid.uuid4())
            ts = float(event.get("timestamp", time.time()))
            key = self._object_key(eid, ts)
            body = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            # Use put_object (simple). Rely on IAM / credentials for access control.
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)
        except Exception:
            logger.exception("S3Backend: failed to write audit event")

    def prune_older_than(self, seconds: int) -> None:
        """
        Delete objects older than cutoff in the configured prefix. This may be expensive for large buckets;
        consider lifecycle policies instead (recommended for production).
        """
        try:
            import datetime

            cutoff_ts = int(time.time() - seconds)
            paginator = self.s3.get_paginator("list_objects_v2")
            prefix = self.prefix + "/"
            to_delete = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # keys use timestamp in filename; try to parse, else fall back to LastModified
                    key = obj["Key"]
                    # Attempt to extract ts from filename: prefix/.../{ts}-{uuid}.json
                    try:
                        basename = key.rsplit("/", 1)[-1]
                        ts_part = basename.split("-", 1)[0]
                        obj_ts = int(ts_part)
                    except Exception:
                        # fallback to LastModified
                        obj_ts = int(obj["LastModified"].timestamp())
                    if obj_ts < cutoff_ts:
                        to_delete.append({"Key": key})
                    # batch delete in groups of 1000
                    if len(to_delete) >= 1000:
                        self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": to_delete})
                        to_delete = []
            if to_delete:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": to_delete})
        except Exception:
            logger.exception("S3Backend: prune failed; consider using bucket lifecycle rules instead")


# instantiate backend
_backend: Optional[AuditBackend] = None
try:
    if AUDIT_BACKEND == "s3":
        if not AUDIT_S3_BUCKET:
            raise RuntimeError("AUDIT_BACKEND=s3 requires AUDIT_S3_BUCKET to be set")
        _backend = S3Backend(bucket=AUDIT_S3_BUCKET, prefix=AUDIT_S3_PREFIX)
    elif AUDIT_BACKEND == "sqlite":
        _backend = SQLiteBackend(db_path=AUDIT_DB_PATH)
    else:
        _backend = FileBackend(path=FILE_LOG_FILE)
except Exception:
    logger.exception("Failed to initialize audit backend; falling back to file backend")
    _backend = FileBackend(path=FILE_LOG_FILE)


def audit_event(event: Dict[str, Any]) -> None:
    """
    Public entrypoint to record an audit event.

    The event dict must be JSON-serializable. To reduce PII leakage, backend implementations should
    ensure only masked snippets are included in the event (caller responsibility).
    """
    # Ensure canonical fields
    if "request_id" not in event:
        event["request_id"] = str(uuid.uuid4())
    if "timestamp" not in event:
        event["timestamp"] = time.time()

    try:
        _backend.write(event)
    except Exception:
        logger.exception("audit_event: backend write failed")


def prune_old(retention_days: Optional[int] = None) -> None:
    """
    Prune events older than retention_days (defaults to AUDIT_RETENTION_DAYS).
    Some backends (S3) recommend lifecycle policies; prune_old attempts deletion but may be expensive.
    """
    days = retention_days if retention_days is not None else AUDIT_RETENTION_DAYS
    seconds = int(days) * 24 * 60 * 60
    try:
        _backend.prune_older_than(seconds)
    except NotImplementedError:
        logger.warning("Audit backend does not support pruning")
    except Exception:
        logger.exception("prune_old failed")
