"""M1 single-agent research loop.

plan -> (search -> fetch -> extract claims/evidence) -> verify claims ->
[if evidence insufficient and budget/rounds remain: supplementary search] ->
write report -> final verify -> [if failed and budget/rounds remain: rewrite] ->
persist report + citations.

"Rounds" is a single counter shared across initial-verification gaps and
final-verification failures, capped at MAX_SUPPLEMENT_ROUNDS (plan.md 178:
these share the same全任务最多 2 轮 budget, not independent limits each).
"""

import logging
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kresearch import storage
from kresearch.agent import prompts
from kresearch.budget.ledger import BudgetExceededError
from kresearch.db.base import async_session_factory
from kresearch.db.models import (
    Citation,
    Claim,
    ClaimEvidence,
    Evidence,
    Report,
    Source,
    SourceSnapshot,
    Task,
    TaskSource,
)
from kresearch.fetch.web import UnsafeURLError, canonicalize_url, contains_suspicious_pattern, fetch_and_extract
from kresearch.llm.client import complete
from kresearch.search.tavily import search as tavily_search
from kresearch.util import safe_json_object as _safe_json

logger = logging.getLogger("kresearch.agent")

MAX_SUPPLEMENT_ROUNDS = 2
MAX_URLS_PER_QUERY = 3
MAX_CLAIMS_FOR_REPORT = 40
MAX_SNAPSHOT_CHARS_FOR_LLM = 8000
JSON_OBJECT = {"type": "json_object"}
CITATION_RE = re.compile(r"\[C(\d+)\]")

ASSUMPTIONS_NOTE_EN = (
    "No clarifying question was asked before running this task; the query was researched "
    "as literally stated, without an additionally assumed time range or region."
)


def _dedupe_and_cap_claims(claims: list[Claim], max_claims: int) -> list[uuid.UUID]:
    """Drop exact-duplicate claim text and cap the rest at max_claims.

    Broad questions can produce 100+ verified claims across many pages with
    no consolidation step (that's the Analysis/Synthesis Agent's job in the
    target design, plan.md 4.1 -- M1 doesn't have one yet). Feeding all of
    them into one write/verify call both makes for an unfocused report and
    risks exceeding even the dynamic max_tokens ceiling. This is a stopgap
    selection by extraction order, not real relevance ranking.
    """
    seen_text: set[str] = set()
    selected: list[uuid.UUID] = []
    for claim in claims:
        key = claim.text.strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        selected.append(claim.id)
        if len(selected) >= max_claims:
            break
    return selected


def _dynamic_max_tokens(n_items: int, per_item: int = 40, base: int = 200, cap: int = 6000) -> int:
    """Scale max_tokens with how many items the model must cover in one JSON
    response — a fixed budget silently truncates (and thus drops) verdicts
    once claim/citation counts grow past what it was sized for.
    """
    return min(cap, base + per_item * max(n_items, 1))


async def _set_status(task_id: uuid.UUID, status: str) -> None:
    async with async_session_factory() as session:
        task = await session.get(Task, task_id)
        task.status = status
        await session.commit()


async def _finish_task(task_id: uuid.UUID, status: str) -> None:
    async with async_session_factory() as session:
        task = await session.get(Task, task_id)
        task.status = status
        task.finished_at = datetime.now(timezone.utc)
        await session.commit()


async def _plan_queries(task_id: uuid.UUID, query: str) -> list[str]:
    async with async_session_factory() as session:
        raw = await complete(
            session, task_id, agent_name="planner",
            messages=prompts.build_plan_queries_prompt(query),
            call_key=f"{task_id}:plan:1", max_tokens=300, response_format=JSON_OBJECT,
        )
    data = _safe_json(raw)
    queries = data.get("queries") or []
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    return queries[:4] or [query]


async def _get_or_create_source(session: AsyncSession, canonical_url: str, title: str) -> Source:
    existing = await session.execute(
        select(Source).where(Source.owner_scope == "public", Source.canonical_url == canonical_url)
    )
    source = existing.scalar_one_or_none()
    if source:
        return source
    source = Source(owner_scope="public", source_type="web", canonical_url=canonical_url, title=title)
    session.add(source)
    await session.commit()
    await session.refresh(source)
    return source


async def _get_or_create_snapshot(session: AsyncSession, source_id: uuid.UUID, text: str) -> SourceSnapshot:
    content_hash = storage.content_hash(text)
    existing = await session.execute(
        select(SourceSnapshot).where(
            SourceSnapshot.source_id == source_id, SourceSnapshot.content_hash == content_hash
        )
    )
    snapshot = existing.scalar_one_or_none()
    if snapshot:
        return snapshot
    snapshot = SourceSnapshot(source_id=source_id, content_hash=content_hash, content_ref="")
    session.add(snapshot)
    await session.flush()
    snapshot.content_ref = storage.save_snapshot(snapshot.id, text)
    await session.commit()
    await session.refresh(snapshot)
    return snapshot


async def _link_task_source(session: AsyncSession, task_id: uuid.UUID, snapshot_id: uuid.UUID, credibility: float) -> None:
    existing = await session.execute(
        select(TaskSource).where(TaskSource.task_id == task_id, TaskSource.snapshot_id == snapshot_id)
    )
    if existing.scalar_one_or_none():
        return
    session.add(TaskSource(task_id=task_id, snapshot_id=snapshot_id, credibility_score=credibility))
    await session.commit()


async def _search_and_fetch(
    task_id: uuid.UUID, sub_query: str, round_no: int, query_idx: int, seen_urls: set[str]
) -> list[tuple[SourceSnapshot, str, str, str]]:
    """Search, fetch each new result, persist source/snapshot rows.

    Returns [(snapshot, text, title, final_url), ...] for newly-seen pages
    only (already-seen canonical URLs are skipped, not re-fetched/re-billed).
    """
    async with async_session_factory() as session:
        results = await tavily_search(
            session, task_id, sub_query,
            call_key=f"{task_id}:search:{round_no}:{query_idx}",
            max_results=MAX_URLS_PER_QUERY,
        )

    collected = []
    for result in results:
        canonical = canonicalize_url(result["url"])
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        try:
            text, final_url = await fetch_and_extract(result["url"])
        except (UnsafeURLError, Exception):
            continue
        if not text or len(text) < 200:
            continue

        credibility = 0.2 if contains_suspicious_pattern(text) else 1.0
        async with async_session_factory() as session:
            source = await _get_or_create_source(session, canonical, result.get("title", ""))
            snapshot = await _get_or_create_snapshot(session, source.id, text)
            await _link_task_source(session, task_id, snapshot.id, credibility)
        collected.append((snapshot, text[:MAX_SNAPSHOT_CHARS_FOR_LLM], result.get("title", ""), final_url))
    return collected


async def _extract_claims(
    task_id: uuid.UUID, query: str, snapshot: SourceSnapshot, text: str, title: str, url: str, call_key: str
) -> list[uuid.UUID]:
    """Extract claims+evidence from one snapshot. Returns new claim IDs.

    A quoted "evidence" that isn't a verbatim substring of the snapshot text
    is dropped rather than stored — it can't be the hallucinated kind of
    citation this project is meant to prevent (plan.md 180).
    """
    async with async_session_factory() as session:
        raw = await complete(
            session, task_id, agent_name="reading",
            messages=prompts.build_extract_prompt(query, title, url, text),
            call_key=call_key, max_tokens=2048, response_format=JSON_OBJECT,
        )
        data = _safe_json(raw)
        claim_ids = []
        for item in data.get("claims", []):
            claim_text = item.get("text")
            if not claim_text:
                continue
            claim = Claim(
                task_id=task_id, text=claim_text,
                claim_type=item.get("claim_type", "fact"), verification_status="pending",
            )
            session.add(claim)
            await session.flush()

            has_evidence = False
            for quote in item.get("quotes", []):
                if not isinstance(quote, str) or quote not in text:
                    continue
                start = text.index(quote)
                evidence = Evidence(
                    task_id=task_id, snapshot_id=snapshot.id, quote=quote,
                    locator={"char_start": start, "char_end": start + len(quote)},
                )
                session.add(evidence)
                await session.flush()
                session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relation="supports"))
                has_evidence = True

            if has_evidence:
                claim_ids.append(claim.id)
            else:
                await session.delete(claim)
        await session.commit()
        return claim_ids


async def _evidence_quotes(session: AsyncSession, claim_id: uuid.UUID) -> list[str]:
    result = await session.execute(
        select(Evidence.quote).join(ClaimEvidence, ClaimEvidence.evidence_id == Evidence.id)
        .where(ClaimEvidence.claim_id == claim_id)
    )
    return list(result.scalars().all())


async def _verify_claims(task_id: uuid.UUID, query: str, claim_ids: list[uuid.UUID], call_key: str) -> None:
    if not claim_ids:
        return
    async with async_session_factory() as session:
        claims = [await session.get(Claim, cid) for cid in claim_ids]
        id_map = {str(i + 1): c.id for i, c in enumerate(claims)}
        payload = [
            {"id": str(i + 1), "text": c.text, "quotes": await _evidence_quotes(session, c.id)}
            for i, c in enumerate(claims)
        ]
        raw = await complete(
            session, task_id, agent_name="critic",
            messages=prompts.build_verify_claims_prompt(query, payload),
            call_key=call_key, max_tokens=_dynamic_max_tokens(len(claims), per_item=25), response_format=JSON_OBJECT,
        )
        data = _safe_json(raw)
        for verdict in data.get("verdicts", []):
            claim_id = id_map.get(verdict.get("id"))
            if not claim_id:
                continue
            claim = await session.get(Claim, claim_id)
            status = verdict.get("status")
            claim.verification_status = status if status in ("supported", "insufficient", "rejected") else "insufficient"
        await session.commit()


async def _write_report(
    task_id: uuid.UUID, query: str, supported_ids: list[uuid.UUID], assumptions: list[str], call_key: str
) -> tuple[str, dict[str, uuid.UUID], dict[uuid.UUID, uuid.UUID]]:
    """Returns (markdown, {marker_number: claim_id}, {claim_id: first_evidence_id})."""
    async with async_session_factory() as session:
        claims = [await session.get(Claim, cid) for cid in supported_ids]
        id_map = {str(i + 1): c.id for i, c in enumerate(claims)}
        claim_to_evidence = {}
        for c in claims:
            quotes_result = await session.execute(
                select(ClaimEvidence.evidence_id).where(ClaimEvidence.claim_id == c.id).limit(1)
            )
            evidence_id = quotes_result.scalar_one_or_none()
            if evidence_id:
                claim_to_evidence[c.id] = evidence_id

        payload = [{"id": str(i + 1), "text": c.text} for i, c in enumerate(claims)]
        markdown = await complete(
            session, task_id, agent_name="writer",
            messages=prompts.build_write_report_prompt(query, payload, assumptions),
            call_key=call_key, max_tokens=_dynamic_max_tokens(len(claims), per_item=80, base=500, cap=6000),
        )
    return markdown, id_map, claim_to_evidence


async def _final_verify(
    task_id: uuid.UUID, supported_ids: list[uuid.UUID], markdown: str, id_map: dict[str, uuid.UUID], call_key: str
) -> tuple[str, list[dict]]:
    async with async_session_factory() as session:
        payload = []
        for marker, claim_id in id_map.items():
            claim = await session.get(Claim, claim_id)
            payload.append({"id": f"C{marker}", "text": claim.text, "quotes": await _evidence_quotes(session, claim_id)})
        raw = await complete(
            session, task_id, agent_name="critic",
            messages=prompts.build_final_verify_prompt(markdown, payload),
            call_key=call_key, max_tokens=_dynamic_max_tokens(len(payload), per_item=40, base=300),
            response_format=JSON_OBJECT,
        )
    data = _safe_json(raw)
    status = data.get("status") if data.get("status") in ("done", "failed") else "failed"
    return status, data.get("issues", [])


async def _persist_report(
    task_id: uuid.UUID, markdown: str, id_map: dict[str, uuid.UUID], claim_to_evidence: dict[uuid.UUID, uuid.UUID],
    version: int, verification_status: str,
) -> str:
    content_ref = storage.save_report(task_id, version, markdown)
    async with async_session_factory() as session:
        report = Report(task_id=task_id, version=version, content_ref=content_ref, verification_status=verification_status)
        session.add(report)
        await session.flush()
        for position, match in enumerate(CITATION_RE.finditer(markdown)):
            claim_id = id_map.get(match.group(1))
            evidence_id = claim_to_evidence.get(claim_id) if claim_id else None
            if claim_id and evidence_id:
                session.add(Citation(report_id=report.id, claim_id=claim_id, evidence_id=evidence_id, position=position))
        await session.commit()
    return content_ref


async def run(task_id: uuid.UUID, query: str) -> dict:
    seen_urls: set[str] = set()
    all_claim_ids: list[uuid.UUID] = []
    supported_ids: list[uuid.UUID] = []
    supplement_rounds_used = 0
    assumptions = [ASSUMPTIONS_NOTE_EN]

    logger.info("task %s: planning", task_id)
    await _set_status(task_id, "planning")
    try:
        sub_queries = await _plan_queries(task_id, query)
    except BudgetExceededError:
        logger.warning("task %s: budget exhausted before planning", task_id)
        await _finish_task(task_id, "failed")
        return {"task_id": str(task_id), "status": "failed", "reason": "budget exhausted before planning"}
    logger.info("task %s: planned %d sub-queries: %s", task_id, len(sub_queries), sub_queries)

    budget_exhausted = False
    round_no = 0
    while True:
        round_no += 1
        logger.info("task %s: round %d executing", task_id, round_no)
        await _set_status(task_id, "executing")

        for idx, sub_query in enumerate(sub_queries):
            if budget_exhausted:
                break
            try:
                collected = await _search_and_fetch(task_id, sub_query, round_no, idx, seen_urls)
                logger.info("task %s: sub-query %r -> %d new pages fetched", task_id, sub_query, len(collected))
            except BudgetExceededError:
                logger.warning("task %s: budget exhausted during search", task_id)
                budget_exhausted = True
                break
            for j, (snapshot, text, title, url) in enumerate(collected):
                if budget_exhausted:
                    break
                try:
                    claim_ids = await _extract_claims(
                        task_id, query, snapshot, text, title, url, call_key=f"{task_id}:extract:{round_no}:{idx}:{j}"
                    )
                    all_claim_ids.extend(claim_ids)
                    logger.info("task %s: extracted %d claims from %s", task_id, len(claim_ids), url)
                except BudgetExceededError:
                    logger.warning("task %s: budget exhausted during extraction", task_id)
                    budget_exhausted = True
                    break

        # Always attempt verification of whatever was collected so far, even if
        # budget ran out mid-search: verification is cheap and claims already
        # extracted deserve a chance to be marked supported before we give up.
        await _set_status(task_id, "verifying")
        try:
            await _verify_claims(task_id, query, all_claim_ids, call_key=f"{task_id}:verify:{round_no}")
        except BudgetExceededError:
            budget_exhausted = True

        async with async_session_factory() as session:
            claims = [await session.get(Claim, cid) for cid in all_claim_ids]
        supported_ids = [c.id for c in claims if c.verification_status == "supported"]
        insufficient = [c for c in claims if c.verification_status != "supported"]
        logger.info(
            "task %s: round %d verified -> %d supported, %d not", task_id, round_no, len(supported_ids), len(insufficient)
        )

        if budget_exhausted or not insufficient or supplement_rounds_used >= MAX_SUPPLEMENT_ROUNDS:
            break
        supplement_rounds_used += 1
        gap = "; ".join(c.text for c in insufficient[:5])
        sub_queries = [f"{query} (additional evidence needed for: {gap})"]
        logger.info("task %s: supplementary round %d for gaps: %s", task_id, supplement_rounds_used, gap)

    if not supported_ids:
        logger.warning("task %s: no supported claims, failing", task_id)
        await _finish_task(task_id, "failed")
        return {"task_id": str(task_id), "status": "failed", "reason": "no claim had verifiable supporting evidence"}

    if len(supported_ids) > MAX_CLAIMS_FOR_REPORT:
        before = len(supported_ids)
        supported_claim_objs = [c for c in claims if c.id in set(supported_ids)]
        supported_ids = _dedupe_and_cap_claims(supported_claim_objs, MAX_CLAIMS_FOR_REPORT)
        logger.info("task %s: capped claims for report from %d to %d (dedup + cap)", task_id, before, len(supported_ids))

    version = 1
    final_status = "failed"
    markdown = ""
    issues: list[dict] = []
    id_map: dict[str, uuid.UUID] = {}
    claim_to_evidence: dict[uuid.UUID, uuid.UUID] = {}
    while True:
        try:
            logger.info("task %s: writing report draft v%d", task_id, version)
            await _set_status(task_id, "synthesizing")
            markdown, id_map, claim_to_evidence = await _write_report(
                task_id, query, supported_ids, assumptions, call_key=f"{task_id}:write:{version}"
            )
            await _set_status(task_id, "verifying")
            final_status, issues = await _final_verify(
                task_id, supported_ids, markdown, id_map, call_key=f"{task_id}:finalverify:{version}"
            )
            logger.info("task %s: draft v%d final verification -> %s (%d issues)", task_id, version, final_status, len(issues))
        except BudgetExceededError:
            logger.warning("task %s: budget exhausted during synthesis/verification", task_id)
            budget_exhausted = True
            final_status = "failed" if not markdown else "partial"
            break

        if final_status == "done" or supplement_rounds_used >= MAX_SUPPLEMENT_ROUNDS:
            break
        supplement_rounds_used += 1
        version += 1

    task_status = "done" if final_status == "done" else "partial"
    content_ref = await _persist_report(task_id, markdown, id_map, claim_to_evidence, version, final_status)
    logger.info("task %s: finished with status=%s report=%s", task_id, task_status, content_ref)
    await _finish_task(task_id, task_status)
    return {
        "task_id": str(task_id),
        "status": task_status,
        "report_path": content_ref,
        "rounds_used": supplement_rounds_used,
        "issues": issues,
    }
