
"""
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    try:
        scheme, token = auth.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")

    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization must use Bearer scheme")

    token = token.strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Empty Bearer token")

    if not JWT_SECRET:
        # Misconfiguration on server side — return generic 500 without revealing secret state
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server authentication misconfigured")

    # Build decode options and claims verification
    options = {"verify_exp": True}
    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options=options,
        )
        # Note: decoded is a dict of claims
        return decoded
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def require_auth(request: Request) -> Optional[Dict]:
    """
    FastAPI dependency. Returns decoded JWT payload (when in JWT mode), an empty dict for valid API key auth,
    or None if AUTH_MODE is 'none'. Raises HTTPException on failure.

    Usage:
        @app.post("/predict")
        def predict(..., _auth=Depends(require_auth)):
            ...
    """
    mode = AUTH_MODE
    if mode == "none":
        return None
    if mode == "api_key":
        ok = _validate_api_key_from_header(request)
        if not ok:
            # Do not reveal whether key was missing vs invalid
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
        return {}
    if mode == "jwt":
        payload = _validate_jwt_from_header(request)
        return payload
    # unreachable due to module-level validation, but keep safe fallback
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid auth configuration")
