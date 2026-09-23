"""Shared task-creation helper used by both the CLI and the HTTP API."""

import hashlib
import json
import uuid

from kresearch.db.base import async_session_factory
from kresearch.db.models import Task


def request_hash(query: str, source_mode: str, budget: float) -> str:
    payload = json.dumps({"query": query, "source_mode": source_mode, "budget": budget}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def create_task(query: str, budget: float, user: str = "local", source_mode: str = "web") -> uuid.UUID:
    async with async_session_factory() as session:
        task = Task(
            user_id=user,
            idempotency_key=str(uuid.uuid4()),
            request_hash=request_hash(query, source_mode, budget),
            query=query,
            source_mode=source_mode,
            budget=budget,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return task.id
