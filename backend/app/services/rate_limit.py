import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


_buckets: dict[str, deque[float]] = defaultdict(deque)


async def rate_limit(request: Request) -> None:
    settings = get_settings()
    key = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _buckets[key]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    bucket.append(now)
