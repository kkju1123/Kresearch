"""FastAPI app: submit a research task, watch progress over SSE, read the
final report. This is deliberately NOT the full M4 async task service
(no Celery/Arq, no outbox, no persisted task_events) -- for a single local
user, `asyncio.create_task` running in this same process is enough, and the
SSE stream is implemented by polling `tasks.status` (already updated by
agent/loop.py's _set_status()) rather than a pub/sub bus. plan.md's fuller
async design is deferred to M4; this is the honest M1-sized version of it.
"""

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select

from kresearch.agent.loop import run as run_loop
from kresearch.budget import ledger
from kresearch.config import settings
from kresearch.db.base import async_session_factory
from kresearch.db.models import Report, Task
from kresearch.tasks import create_task

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
TERMINAL_STATUSES = {"done", "partial", "failed", "cancelled", "timed_out"}
POLL_INTERVAL_SECONDS = 1.0

app = FastAPI(title="KResearch")

# Keep references to background tasks so they aren't garbage-collected
# mid-run (a well-known asyncio.create_task gotcha).
_background_tasks: dict[uuid.UUID, asyncio.Task] = {}


class CreateTaskRequest(BaseModel):
    query: str
    budget: float = settings.default_task_budget_usd


class CreateTaskResponse(BaseModel):
    task_id: uuid.UUID


@app.post("/research/tasks", response_model=CreateTaskResponse, status_code=202)
async def submit_task(body: CreateTaskRequest) -> CreateTaskResponse:
    task_id = await create_task(body.query, body.budget)
    coro = run_loop(task_id, body.query)
    background = asyncio.create_task(coro)
    _background_tasks[task_id] = background
    background.add_done_callback(lambda _: _background_tasks.pop(task_id, None))
    return CreateTaskResponse(task_id=task_id)


async def _get_task_or_404(task_id: uuid.UUID) -> Task:
    async with async_session_factory() as session:
        task = await session.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/research/tasks/{task_id}")
async def get_task(task_id: uuid.UUID) -> dict:
    task = await _get_task_or_404(task_id)
    async with async_session_factory() as session:
        remaining = await ledger.remaining_budget(session, task_id)
    spent = float(task.budget) - remaining

    result = {
        "task_id": str(task_id),
        "status": task.status,
        "budget": float(task.budget),
        "spent": round(spent, 6),
    }

    if task.status in TERMINAL_STATUSES:
        async with async_session_factory() as session:
            report = (
                await session.execute(
                    select(Report).where(Report.task_id == task_id).order_by(Report.version.desc()).limit(1)
                )
            ).scalar_one_or_none()
        if report is not None:
            result["report_markdown"] = Path(report.content_ref).read_text(encoding="utf-8")
            result["verification_status"] = report.verification_status

    return result


@app.get("/research/tasks/{task_id}/stream")
async def stream_task(task_id: uuid.UUID) -> StreamingResponse:
    await _get_task_or_404(task_id)

    async def event_generator():
        last_status = None
        while True:
            async with async_session_factory() as session:
                task = await session.get(Task, task_id)
            if task is None:
                yield f"event: error\ndata: {json.dumps({'error': 'task not found'})}\n\n"
                return

            if task.status != last_status:
                last_status = task.status
                async with async_session_factory() as session:
                    remaining = await ledger.remaining_budget(session, task_id)
                payload = {
                    "status": task.status,
                    "spent": round(float(task.budget) - remaining, 6),
                    "budget": float(task.budget),
                }
                yield f"data: {json.dumps(payload)}\n\n"

            if task.status in TERMINAL_STATUSES:
                return
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Registered last: falls back to serving the static frontend for any path
# not matched by the API routes above (and "/" -> static/index.html).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
