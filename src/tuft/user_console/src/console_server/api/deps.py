from fastapi import Header, HTTPException


def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(status_code=400, detail="Missing X-API-Key header")
    return x_api_key
