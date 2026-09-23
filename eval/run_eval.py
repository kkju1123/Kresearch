"""M1 baseline evaluation harness (plan.md 6.3).

For each question in questions.yaml: run the research loop, then score it
two independent ways:
  1. Programmatic citation check -- re-derived from the DB and the actual
     saved snapshot files, not trusting the pipeline's own bookkeeping.
  2. LLM-as-Judge -- a model call that never touched this task's research
     process, scoring accuracy / evidence_support / coverage / logical
     consistency against the question's required_points.

Usage: uv run python eval/run_eval.py [question_id ...]
With no arguments, runs every question in questions.yaml. With one or more
question ids (e.g. `q03 q04`), runs only those -- useful for re-running a
subset after a fix without re-spending on questions that already passed.
Writes a timestamped JSON + Markdown summary to eval/results/.
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import select

from kresearch.agent import loop, prompts
from kresearch.budget import ledger
from kresearch.db.base import async_session_factory
from kresearch.db.models import Citation, Claim, Evidence, Report, SourceSnapshot, Task
from kresearch.llm.client import complete
from kresearch.storage import load_snapshot
from kresearch.util import safe_json_object

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BUDGET = 0.3
JUDGE_BUDGET_USD = 1.0
JUDGE_MAX_TOKENS = 500


async def _create_task(query: str, budget: float) -> uuid.UUID:
    async with async_session_factory() as session:
        task = Task(idempotency_key=str(uuid.uuid4()), request_hash=f"eval:{query}", query=query, budget=budget)
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return task.id


async def _check_citations(report_id: uuid.UUID) -> tuple[int, int]:
    """Independently re-verify: does every citation's evidence quote actually
    appear in the snapshot file it claims to come from?
    """
    async with async_session_factory() as session:
        citations = (await session.execute(select(Citation).where(Citation.report_id == report_id))).scalars().all()
        valid = 0
        for c in citations:
            evidence = await session.get(Evidence, c.evidence_id)
            claim = await session.get(Claim, c.claim_id)
            if not evidence or not claim:
                continue
            snapshot = await session.get(SourceSnapshot, evidence.snapshot_id)
            if not snapshot:
                continue
            try:
                text = load_snapshot(snapshot.content_ref)
            except FileNotFoundError:
                continue
            if evidence.quote in text:
                valid += 1
        return valid, len(citations)


async def _judge(
    judge_task_id: uuid.UUID, query: str, required_points: list[str],
    report_id: uuid.UUID, report_markdown: str, call_key: str,
) -> dict:
    async with async_session_factory() as session:
        citations = (await session.execute(select(Citation).where(Citation.report_id == report_id))).scalars().all()
        evidence_payload = []
        for c in citations:
            claim = await session.get(Claim, c.claim_id)
            evidence = await session.get(Evidence, c.evidence_id)
            if claim and evidence:
                evidence_payload.append({"claim": claim.text, "quote": evidence.quote})

        raw = await complete(
            session, judge_task_id, agent_name="judge",
            messages=prompts.build_judge_prompt(query, required_points, report_markdown, evidence_payload),
            call_key=call_key, max_tokens=JUDGE_MAX_TOKENS, response_format={"type": "json_object"},
        )
    return safe_json_object(raw)


async def run_all(only_ids: set[str] | None = None) -> list[dict]:
    questions = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if only_ids:
        questions = [q for q in questions if q["id"] in only_ids]
    judge_task_id = await _create_task("__eval_judge__", JUDGE_BUDGET_USD)

    results = []
    for q in questions:
        print(f"[{q['id']}] {q['query']}")
        budget = q.get("budget", DEFAULT_BUDGET)
        task_id = await _create_task(q["query"], budget)

        started = time.monotonic()
        outcome = await loop.run(task_id, q["query"])
        elapsed = time.monotonic() - started

        async with async_session_factory() as session:
            spent = budget - await ledger.remaining_budget(session, task_id)

        row = {
            "id": q["id"],
            "type": q["type"],
            "query": q["query"],
            "status": outcome["status"],
            "elapsed_seconds": round(elapsed, 1),
            "cost_usd": round(spent, 6),
            "rounds_used": outcome.get("rounds_used"),
            "reason": outcome.get("reason"),
            "citation_valid": 0,
            "citation_total": 0,
            "citation_validity_rate": None,
            "judge": None,
        }

        report_path = outcome.get("report_path")
        if report_path:
            report_markdown = Path(report_path).read_text(encoding="utf-8")
            async with async_session_factory() as session:
                report = (
                    await session.execute(select(Report).where(Report.content_ref == report_path))
                ).scalar_one()
            valid, total = await _check_citations(report.id)
            row["citation_valid"] = valid
            row["citation_total"] = total
            row["citation_validity_rate"] = round(valid / total, 3) if total else None
            row["judge"] = await _judge(
                judge_task_id, q["query"], q.get("required_points", []), report.id, report_markdown,
                call_key=f"{judge_task_id}:judge:{q['id']}",
            )

        print(
            f"  -> {row['status']}, cost=${row['cost_usd']}, "
            f"citations {row['citation_valid']}/{row['citation_total']}"
        )
        results.append(row)

    return results


def _summarize(results: list[dict]) -> str:
    lines = [
        "| id | type | status | cost($) | time(s) | citations | accuracy | coverage | evidence | logic |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r.get("judge") or {}
        citation_str = f"{r['citation_valid']}/{r['citation_total']}" if r["citation_total"] else "-"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['status']} | {r['cost_usd']} | {r['elapsed_seconds']} | "
            f"{citation_str} | {j.get('accuracy', '-')} | {j.get('coverage', '-')} | "
            f"{j.get('evidence_support', '-')} | {j.get('logical_consistency', '-')} |"
        )

    total_cost = sum(r["cost_usd"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    valid_citations = sum(r["citation_valid"] for r in results)
    total_citations = sum(r["citation_total"] for r in results)
    summary = [
        f"Tasks: {len(results)}, done={sum(1 for r in results if r['status'] == 'done')}, "
        f"partial={sum(1 for r in results if r['status'] == 'partial')}, "
        f"failed={sum(1 for r in results if r['status'] == 'failed')}",
        f"Total cost: ${total_cost:.4f}, total time: {total_time:.0f}s",
        "Citation validity: "
        + (f"{valid_citations}/{total_citations} ({valid_citations / total_citations:.1%})" if total_citations else "n/a"),
    ]
    return "\n".join(lines) + "\n\n" + "\n".join(summary)


def main() -> None:
    only_ids = set(sys.argv[1:]) or None
    results = asyncio.run(run_all(only_ids))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    summary_md = _summarize(results)
    (RESULTS_DIR / f"{stamp}.md").write_text(summary_md, encoding="utf-8")

    print("\n" + summary_md)
    print(f"\nSaved: eval/results/{stamp}.json and eval/results/{stamp}.md")


if __name__ == "__main__":
    main()
