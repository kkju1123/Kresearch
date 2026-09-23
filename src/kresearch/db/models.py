import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Index, Numeric, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kresearch.db.base import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Task(Base):
    """Research task. Status values match plan.md 3.3's unified state machine:
    pending -> planning -> executing -> synthesizing -> verifying -> done,
    with terminal states partial / failed / cancelled / timed_out.
    """

    __tablename__ = "tasks"
    __table_args__ = (UniqueConstraint("user_id", "idempotency_key", name="uq_tasks_user_idem_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="local")
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_base_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    source_mode: Mapped[str] = mapped_column(Text, nullable=False, default="web")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    budget: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    plan_version: Mapped[int] = mapped_column(nullable=False, default=1)
    checkpoint: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Source(Base):
    """Source identity, not owned by any single task (plan.md 3.2, line 80)."""

    __tablename__ = "sources"
    __table_args__ = (
        Index(
            "uq_sources_owner_url",
            "owner_scope",
            "canonical_url",
            unique=True,
            postgresql_where=Text("canonical_url IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    owner_scope: Mapped[str] = mapped_column(Text, nullable=False, default="public")
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="web")
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)


class SourceSnapshot(Base):
    """Immutable content snapshot of a source (plan.md 3.2, line 81)."""

    __tablename__ = "source_snapshots"
    __table_args__ = (UniqueConstraint("source_id", "content_hash", name="uq_snapshot_source_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id"), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    content_ref: Mapped[str] = mapped_column(Text, nullable=False)
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(nullable=False, default=_now)


class TaskSource(Base):
    """Many-to-many: task <-> source snapshot, with task-scoped relevance (plan.md 3.2, line 82)."""

    __tablename__ = "task_sources"

    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    snapshot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_snapshots.id"), primary_key=True)
    relevance_score: Mapped[float | None] = mapped_column(nullable=True)
    credibility_score: Mapped[float | None] = mapped_column(nullable=True)


class Evidence(Base):
    """Original-text evidence with a locator into the snapshot (plan.md 3.2, line 83)."""

    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    snapshot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_snapshots.id"), nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    locator: Mapped[dict] = mapped_column(JSONB, nullable=False)


class Claim(Base):
    """Shared fact-pool entry (plan.md 3.2, line 84)."""

    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(Text, nullable=False, default="fact")
    verification_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")


class ClaimEvidence(Base):
    """Claim <-> evidence mapping; both must belong to the same task (plan.md 3.2, line 85)."""

    __tablename__ = "claim_evidence"

    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), primary_key=True)
    evidence_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evidence.id"), primary_key=True)
    relation: Mapped[str] = mapped_column(Text, nullable=False, default="supports")
    verification_result: Mapped[str | None] = mapped_column(Text, nullable=True)


class Report(Base):
    """Versioned draft/final report (plan.md 3.2, line 86)."""

    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    content_ref: Mapped[str] = mapped_column(Text, nullable=False)
    verification_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")


class Citation(Base):
    """Report position -> claim + original evidence mapping (plan.md 3.2, line 87)."""

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id"), nullable=False)
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evidence.id"), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)


class ToolCall(Base):
    """Budget ledger entry for a single tool/LLM invocation (plan.md 3.2, line 89)."""

    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    subtask_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    attempt_no: Mapped[int] = mapped_column(nullable=False, default=1)
    call_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="reserved")
    reserved_cost: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    actual_cost: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    provider_request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_now)


class Message(Base):
    """Full agent/LLM interaction log for debugging and replay (plan.md 3.2, line 91)."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_now)
