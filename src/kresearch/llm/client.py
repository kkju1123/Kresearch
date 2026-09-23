"""DeepSeek (OpenAI-compatible) chat completion client.

Every call goes through the budget ledger: reserve an upper-bound estimate
before calling, settle with the real cost (from the response's usage) after.
Every call's prompt/response is also written to `messages` for debugging and
replay (plan.md 3.2, `messages` table).
"""

import uuid

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from kresearch.budget import ledger
from kresearch.config import settings
from kresearch.db.models import Message

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    return _client

# Rough per-token USD pricing, used only for budget estimation/accounting.
# DeepSeek's published pricing changes independently of this code; treat this
# as a conservative estimate, not a billing source of truth.
PRICE_PER_INPUT_TOKEN = 0.28 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.42 / 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICE_PER_INPUT_TOKEN + output_tokens * PRICE_PER_OUTPUT_TOKEN


def _rough_token_count(text: str) -> int:
    # No DeepSeek tokenizer wired up for M1; 1 token ~= 4 chars is a
    # deliberately conservative over-estimate for the pre-call budget check.
    return max(len(text) // 4, 1)


async def _log_message(session: AsyncSession, task_id: uuid.UUID, role: str, content: str, agent_name: str) -> None:
    session.add(Message(task_id=task_id, role=role, content=content, agent_name=agent_name))
    await session.commit()


async def complete(
    session: AsyncSession,
    task_id: uuid.UUID,
    agent_name: str,
    messages: list[dict],
    call_key: str,
    max_tokens: int = 1024,
    subtask_id: uuid.UUID | None = None,
    attempt_no: int = 1,
    response_format: dict | None = None,
) -> str:
    """Run one chat completion under budget control. Returns the reply text."""
    input_text = "\n".join(m["content"] for m in messages)
    estimated = estimate_cost(_rough_token_count(input_text), max_tokens)

    call = await ledger.reserve(session, task_id, call_key, estimated, subtask_id, attempt_no)
    await _log_message(session, task_id, "user", input_text, agent_name)

    kwargs: dict = dict(model=settings.deepseek_model, messages=messages, max_tokens=max_tokens)
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        response = await _get_client().chat.completions.create(**kwargs)
    except Exception:
        await ledger.settle(session, call, actual_cost=0.0, status="failed")
        raise

    usage = response.usage
    actual_cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
    await ledger.settle(session, call, actual_cost=actual_cost, status="settled", provider_request_id=response.id)

    content = response.choices[0].message.content or ""
    await _log_message(session, task_id, "assistant", content, agent_name)
    return content
