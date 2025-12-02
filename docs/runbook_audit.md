 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
```markdown
   - For file/sqlite: configure nightly backups to an immutable store (S3 archival bucket).
   - For S3: use versioning if you need point-in-time recovery, and lifecycle to transition to Glacier/Deep Archive.

5) Incident response (on flagged safety event)
   - On safety flag, application emits audit event. Typical on-call flow:
     1. Pager/Alert triggers (based on Prometheus alert for excessive safety flags).
     2. Run: grep -n 'flagged":true' logs/safety_audit.log  OR query DB OR query S3 prefix for recent flagged events.
     3. Triage: review event text_snippet, model_version, and details. Do NOT expose full text to unauthorized reviewers.
     4. If human-review needed, move event to a restricted review system (ticket) and tag with request_id.
     5. If incident is a false positive, record decision and update safety rules/model versioning.
     6. If incident is a true policy violation, follow escalation contacts defined in team runbook (legal/compliance/security).

6) Auditing & Compliance
   - Periodically (quarterly) rotate and rekey API keys used for audit ingestion or client enrollment.
   - Maintain an access roster for who can read audit logs; audit who accessed logs using object-store access logs or host auth logs.

Security considerations
- Avoid storing unmasked PII in audit events. The caller is responsible for masking; backends will persist what they receive.
- Use TLS (HTTPS) between services and S3. Use IAM roles rather than long-lived credentials when possible.
- For high-sensitivity environments prefer S3 server-side encryption with CMK, and restrict access via IAM conditions.

Troubleshooting
- "audit_event: backend write failed" in app logs:
  - Check backend connectivity (S3 permissions, disk full, DB locked).
  - For SQLite, ensure the DB file is writable by the service user and not on NFS without proper locking.
- Prune errors:
  - For S3: consider setting lifecycle rules instead; prune script may hit API rate limits for large buckets.
  - For SQLite/file: ensure sufficient disk space to write temporary files during pruning.

Appendix: Example S3 Lifecycle (recommended)
- Transition to STANDARD_IA after 30 days
- Transition to GLACIER after 365 days
- Expire (delete) after 3 years

Example lifecycle XML (AWS console / bucket policy):
- See AWS docs: Use lifecycle rule with Prefix = aegis/audit/ and Status = Enabled.

Change log
- 2025-12-01: Added configurable audit backends and prune CLI; runbook created.
```
