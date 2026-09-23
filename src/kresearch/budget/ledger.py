"""Budget ledger: reserve cost before a paid call, settle it after.

Every LLM/search call must go through reserve() -> (do the call) -> settle().
reserve() locks the task row and checks committed spend (settled actual costs +
still-open reservations, which count against budget at their upper-bound
estimate) before allowing a new reservation. This mirrors plan.md 3.5/116:
budget is enforced atomically per task, and an unresolved call's cost is not
released until its outcome is known.
"""

import uuid

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kresearch.db.models import Task, ToolCall

OPEN_STATUSES = ("reserved", "unknown")


class BudgetExceededError(Exception):
    def __init__(self, task_id: uuid.UUID, committed: float, requested: float, budget: float):
        self.task_id = task_id
        self.committed = committed
        self.requested = requested
        self.budget = budget
        super().__init__(
            f"task {task_id}: committed {committed:.6f} + requested {requested:.6f} "
            f"exceeds budget {budget:.6f}"
        )


async def _committed_spend(session: AsyncSession, task_id: uuid.UUID) -> float:
    cost_expr = case(
        (ToolCall.status == "settled", ToolCall.actual_cost),
        (ToolCall.status.in_(OPEN_STATUSES), ToolCall.reserved_cost),
        else_=0,
    )
    result = await session.execute(
        select(func.coalesce(func.sum(cost_expr), 0)).where(ToolCall.task_id == task_id)
    )
    return float(result.scalar_one())


async def reserve(
    session: AsyncSession,
    task_id: uuid.UUID,
    call_key: str,
    estimated_cost: float,
    subtask_id: uuid.UUID | None = None,
    attempt_no: int = 1,
) -> ToolCall:
    """Atomically check budget and create a 'reserved' ToolCall row.

    Raises BudgetExceededError without writing anything if the task's budget
    would be exceeded. Caller must not perform the paid call until this
    returns successfully.
    """
    task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update())).scalar_one()
    committed = await _committed_spend(session, task_id)
    budget = float(task.budget)
    if committed + estimated_cost > budget:
        raise BudgetExceededError(task_id, committed, estimated_cost, budget)

    call = ToolCall(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt_no=attempt_no,
        call_key=call_key,
        status="reserved",
        reserved_cost=estimated_cost,
    )
    session.add(call)
    await session.commit()
    await session.refresh(call)
    return call


async def settle(
    session: AsyncSession,
    call: ToolCall,
    actual_cost: float,
    status: str = "settled",
    provider_request_id: str | None = None,
    result_ref: str | None = None,
) -> None:
    """Resolve a reserved call: status is one of settled/failed/unknown.

    failed with actual_cost=0 releases the reservation entirely; unknown keeps
    reserved_cost counted against budget (plan.md line 103: at-least-once,
    not exactly-once, so an ambiguous outcome stays charged at its upper bound).
    """
    call.status = status
    call.actual_cost = actual_cost
    call.provider_request_id = provider_request_id
    call.result_ref = result_ref
    await session.commit()


async def remaining_budget(session: AsyncSession, task_id: uuid.UUID) -> float:
    task = (await session.execute(select(Task).where(Task.id == task_id))).scalar_one()
    committed = await _committed_spend(session, task_id)
    return float(task.budget) - committed
