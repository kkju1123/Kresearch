"""Tavily web search client, wired through the budget ledger.

Tavily bills in API credits, not directly in USD; FLAT_COST_PER_SEARCH is a
conservative USD-equivalent placeholder purely for this project's shared
budget accounting, not a real invoice figure.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from tavily import AsyncTavilyClient

from kresearch.budget import ledger
from kresearch.config import settings

_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _client

FLAT_COST_PER_SEARCH = 0.008


async def search(
    session: AsyncSession,
    task_id: uuid.UUID,
    query: str,
    call_key: str,
    max_results: int = 5,
    subtask_id: uuid.UUID | None = None,
    attempt_no: int = 1,
) -> list[dict]:
    """Run one Tavily search under budget control.

    Returns a list of {"url", "title", "snippet"} dicts.
    """
    call = await ledger.reserve(session, task_id, call_key, FLAT_COST_PER_SEARCH, subtask_id, attempt_no)

    try:
        response = await _get_client().search(query=query, max_results=max_results)
    except Exception:
        await ledger.settle(session, call, actual_cost=0.0, status="failed")
        raise

    await ledger.settle(session, call, actual_cost=FLAT_COST_PER_SEARCH, status="settled")

    return [
        {
            "url": r["url"],
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
