import pytest
from sqlalchemy import delete

from kresearch.budget import ledger
from kresearch.budget.ledger import BudgetExceededError
from kresearch.db.base import async_session_factory
from kresearch.db.models import Task, ToolCall


@pytest.fixture
async def task():
    async with async_session_factory() as session:
        t = Task(idempotency_key=f"test-{id(object())}", request_hash="h", query="q", budget=1.0)
        session.add(t)
        await session.commit()
        await session.refresh(t)
        yield t
        async with async_session_factory() as cleanup:
            await cleanup.execute(delete(ToolCall).where(ToolCall.task_id == t.id))
            await cleanup.execute(delete(Task).where(Task.id == t.id))
            await cleanup.commit()


@pytest.mark.asyncio
async def test_reserve_then_settle_reduces_remaining_budget(task):
    async with async_session_factory() as session:
        call = await ledger.reserve(session, task.id, "call-1", estimated_cost=0.3)
        remaining = await ledger.remaining_budget(session, task.id)
        assert remaining == pytest.approx(0.7)

        await ledger.settle(session, call, actual_cost=0.2, status="settled")
        remaining = await ledger.remaining_budget(session, task.id)
        assert remaining == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_reserve_beyond_budget_raises_and_writes_nothing(task):
    async with async_session_factory() as session:
        await ledger.reserve(session, task.id, "call-1", estimated_cost=0.95)

    async with async_session_factory() as session:
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(session, task.id, "call-2", estimated_cost=0.1)

        remaining = await ledger.remaining_budget(session, task.id)
        assert remaining == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_unknown_outcome_still_counts_against_budget(task):
    async with async_session_factory() as session:
        call = await ledger.reserve(session, task.id, "call-1", estimated_cost=0.4)
        await ledger.settle(session, call, actual_cost=0.0, status="unknown")

        remaining = await ledger.remaining_budget(session, task.id)
        assert remaining == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_failed_call_releases_the_reservation(task):
    async with async_session_factory() as session:
        call = await ledger.reserve(session, task.id, "call-1", estimated_cost=0.4)
        await ledger.settle(session, call, actual_cost=0.0, status="failed")

        remaining = await ledger.remaining_budget(session, task.id)
        assert remaining == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_duplicate_call_key_is_rejected(task):
    async with async_session_factory() as session:
        await ledger.reserve(session, task.id, "same-key", estimated_cost=0.1)

    with pytest.raises(Exception):
        async with async_session_factory() as session:
            await ledger.reserve(session, task.id, "same-key", estimated_cost=0.1)
