"""
InsureAI — API Key Authentication
===================================
Simple static API key guard for registry mutation endpoints.
Read-only endpoints remain public.

Usage:
  Add `Depends(require_api_key)` to any protected endpoint.
  Set API_KEY in .env (or environment) before deployment.
  Pass the key as: X-API-Key: <your-key>
"""

from fastapi import Header, HTTPException, status
from backend.core.settings import settings


async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Dependency: validates X-API-Key header against the configured API_KEY.
    Raises 403 on mismatch or missing header.
    Returns the validated key on success.
    """
    if not settings.API_KEY or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Registry mutation endpoints require X-API-Key header.",
        )
    return x_api_key
