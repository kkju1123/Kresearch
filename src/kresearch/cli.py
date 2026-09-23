import asyncio
import hashlib
import json
import logging
import uuid

import typer

from kresearch.agent.loop import run as run_loop
from kresearch.budget import ledger
from kresearch.config import settings
from kresearch.db.base import async_session_factory
from kresearch.db.models import Task

app = typer.Typer(add_completion=False)


def _request_hash(query: str, source_mode: str, budget: float) -> str:
    payload = json.dumps({"query": query, "source_mode": source_mode, "budget": budget}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@app.command("research")
def research(
    query: str = typer.Argument(..., help="The research question to investigate"),
    budget: float = typer.Option(settings.default_task_budget_usd, "--budget", help="Max USD spend for this task"),
    user: str = typer.Option("local", "--user", help="User id (single-user M1 default: 'local')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print phase-by-phase progress logs"),
) -> None:
    """Run one research task end-to-end and print the resulting report."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_research(query=query, budget=budget, user=user))


async def _research(query: str, budget: float, user: str) -> None:
    async with async_session_factory() as session:
        task = Task(
            user_id=user,
            idempotency_key=str(uuid.uuid4()),
            request_hash=_request_hash(query, "web", budget),
            query=query,
            budget=budget,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)
        task_id = task.id

    typer.echo(f"Task {task_id} started (budget=${budget:.4f})")

    result = await run_loop(task_id, query)

    typer.echo(f"\nStatus: {result['status']}")
    if reason := result.get("reason"):
        typer.echo(f"Reason: {reason}")
    if report_path := result.get("report_path"):
        typer.echo(f"Report: {report_path}")
    if (rounds := result.get("rounds_used")) is not None:
        typer.echo(f"Supplement rounds used: {rounds}")
    if result["status"] == "partial" and (issues := result.get("issues")):
        typer.echo("Unresolved issues (report delivered anyway, marked partial):")
        for issue in issues:
            typer.echo(f"  - [{issue.get('citation', '?')}] {issue.get('problem', '')}")

    async with async_session_factory() as session:
        remaining = await ledger.remaining_budget(session, task_id)
    spent = budget - remaining
    typer.echo(f"Spent: ${spent:.6f} / ${budget:.4f} (remaining ${remaining:.6f})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
