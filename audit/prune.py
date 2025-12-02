"""
CLI script to run periodic pruning of audit events.

Intended usage (cron or systemd timer):
- python -m aegis_multimodal_ai_system.audit.prune --days 90

When using S3 in production prefer S3 Lifecycle policies (cheaper, more reliable). This script is useful for
SQLite and file backends or as a secondary prune utility.
"""
import argparse
import logging
import sys

from .audit_logger import prune_old

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="Retention in days (overrides AUDIT_RETENTION_DAYS)")
    args = parser.parse_args()
    try:
        prune_old(retention_days=args.days)
    except Exception as e:
        logger.exception("Prune job failed: %s", e)
        sys.exit(2)
    logger.info("Prune job completed")


if __name__ == "__main__":
    main()
