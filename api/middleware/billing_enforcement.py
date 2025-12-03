  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
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
"""
Billing enforcement middleware.

Behavior:
- Reads tenant id from (in order):
  1) request.state.tenant_id (set by auth)
  2) request.state._token_payload["tenant"] (if auth stored token payload)
  3) X-Tenant-ID header (convenience for testing / internal services)
- Looks up billing account. If billing_suspended is True -> return 402 Payment Required with a
  billing_portal link in the response body.
- Sets request.state.billing_dunning_level = int(dunning_level) (0 if not present) for rate-limiter
  to use for soft enforcement.
- Lightweight and fails-open in case of DB issues (logs error, allows request). If you prefer fail-closed,
  change behavior.
"""

import logging
import traceback
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.db import SessionLocal
from api.models import BillingAccount

logger = logging.getLogger("aegis.billing_enforcement")


class BillingEnforcementMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, billing_portal_url: str = "/v1/billing/portal"):
        super().__init__(app)
        self.billing_portal_url = billing_portal_url

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        tenant_id = None
        try:
            # 1) explicit state set by auth middleware
            tenant_id = getattr(request.state, "tenant_id", None)
            if not tenant_id:
                token_payload = getattr(request.state, "_token_payload", None)
                if token_payload and isinstance(token_payload, dict):
                    tenant_id = token_payload.get("tenant")
            # 2) fallback: header (useful for internal/test callers)
            if not tenant_id:
                tenant_id = request.headers.get("X-Tenant-ID")

            if tenant_id:
                session = SessionLocal()
                try:
                    ba = session.query(BillingAccount).filter_by(tenant_id=str(tenant_id)).one_or_none()
                    if ba:
                        # expose dunning_level for downstream components (rate limiter)
                        try:
                            request.state.billing_dunning_level = int(getattr(ba, "dunning_level", 0) or 0)
                        except Exception:
                            request.state.billing_dunning_level = 0

                        if getattr(ba, "billing_suspended", False):
                            # soft response describing next steps: link to billing portal
                            payload = {
                                "code": 402,
                                "message": "Payment required: tenant billing suspended",
                                "billing_portal": self.billing_portal_url,
                                "dunning_level": request.state.billing_dunning_level,
                            }
                            return JSONResponse(status_code=402, content=payload)
                finally:
                    try:
                        session.close()
                    except Exception:
                        pass
        except Exception:
            # don't break API availability because of billing lookup issues; log and proceed (fail-open)
            logger.error("Billing enforcement check failed: %s\n%s", traceback.format_exc(), tenant_id)
            # Optionally: return a 503 here to be strict. We choose fail-open by default.
        # proceed to next middleware / handler
        return await call_next(request)
