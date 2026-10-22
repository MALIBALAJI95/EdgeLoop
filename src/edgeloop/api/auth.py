"""Authentication utilities for EdgeLoop API."""

import os
from fastapi import Header, HTTPException, status
from edgeloop.common.config import get_config


def verify_admin_token(x_edgeloop_token: str = Header(None)) -> str:
    """Validate admin authorization token for state-changing endpoints."""
    config = get_config()
    dev_mode = os.getenv("EDGELOOP_DEV_MODE", "true").lower() == "true"
    expected_token = os.getenv("EDGELOOP_API_TOKEN", config.get("api.auth_token", "edgeloop-dev-secret-token"))

    # In dev mode, if no token header is provided, accept default dev token with warning
    if not x_edgeloop_token:
        if dev_mode:
            return "dev-mode-user"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required authorization header 'X-EdgeLoop-Token'"
        )

    if x_edgeloop_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid EdgeLoop authorization token."
        )

    return "authorized-admin"
