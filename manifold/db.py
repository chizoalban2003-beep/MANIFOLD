"""Async database persistence layer for MANIFOLD state.

Provides :class:`ManifoldDB` — an async-compatible layer backed by SQLite
(via ``aiosqlite``) by default.  Set ``MANIFOLD_DB_URL`` to a
``postgresql+asyncpg://...`` connection string to switch to PostgreSQL.

Schema
------
Three tables are created on first connect:

* ``task_outcomes`` — completed tasks with action, cost, risk, asset.
* ``tool_reputation`` — time-series reputation scores per tool.
* ``gossip_events``  — gossip propagation events (source → target).

Design constraints
------------------
* **Zero mandatory core deps** — ``aiosqlite`` is listed as an optional
  extra (``db``).  PostgreSQL support requires ``asyncpg`` via the optional
  ``db-pg`` extra.  Neither is imported at module-level; imports are
  deferred to :meth:`ManifoldDB.connect`.
* **Async-compatible** — all public methods are ``async def``.
* **In-memory testing** — pass ``":memory:"`` as *db_url* (SQLite only) to
  get a fully isolated in-process database suitable for unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_TASK_OUTCOMES = """
CREATE TABLE IF NOT EXISTS task_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_hash TEXT    NOT NULL,
    domain      TEXT    NOT NULL,
    stakes      REAL,
    action      TEXT    NOT NULL,
    cost        REAL,
    risk        REAL,
    asset       REAL,
    outcome     TEXT,
    created_at  REAL    NOT NULL
);
"""

_CREATE_TOOL_REPUTATION = """
CREATE TABLE IF NOT EXISTS tool_reputation (
    tool_name   TEXT NOT NULL,
    score       REAL NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tool_name, recorded_at)
);
"""

_CREATE_GOSSIP_EVENTS = """
CREATE TABLE IF NOT EXISTS gossip_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    risk_vector TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""

# Postgres-compatible DDL differs only in AUTOINCREMENT → SERIAL / GENERATED
_CREATE_TASK_OUTCOMES_PG = """
CREATE TABLE IF NOT EXISTS task_outcomes (
    id          SERIAL PRIMARY KEY,
    prompt_hash TEXT    NOT NULL,
    domain      TEXT    NOT NULL,
    stakes      REAL,
    action      TEXT    NOT NULL,
    cost        REAL,
    risk        REAL,
    asset       REAL,
    outcome     TEXT,
    created_at  REAL    NOT NULL
);
"""

_CREATE_GOSSIP_EVENTS_PG = """
CREATE TABLE IF NOT EXISTS gossip_events (
    id          SERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    risk_vector TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# ManifoldDB
# ---------------------------------------------------------------------------


class ManifoldDB:
    """Async-compatible database layer for MANIFOLD state.

    Parameters
    ----------
    db_url:
        Database connection string.  Defaults to the ``MANIFOLD_DB_URL``
        environment variable, or ``"sqlite:///manifold.db"`` if unset.
        Pass ``":memory:"`` to use an ephemeral in-memory SQLite database.

    Example
    -------
    ::

        db = ManifoldDB(":memory:")
        await db.connect()
        await db.save_task_outcome(task, action="escalate", outcome={})
        stats = await db.get_domain_stats("billing")
        await db.close()

    Or as an async context manager::

        async with ManifoldDB(":memory:") as db:
            await db.save_task_outcome(task, action="answer", outcome={})
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url: str = (
            db_url
            or os.environ.get("MANIFOLD_DB_URL", "sqlite:///manifold.db")
        )
        self._is_postgres: bool = self._db_url.startswith("postgresql")
        self._conn: Any = None  # aiosqlite.Connection or asyncpg.Connection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the database connection and initialise the schema."""
        if self._is_postgres:
            await self._connect_postgres()
        else:
            await self._connect_sqlite()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "ManifoldDB":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Private — SQLite backend
    # ------------------------------------------------------------------

    async def _connect_sqlite(self) -> None:
        try:
            import aiosqlite  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required for SQLite persistence. "
                "Install it with: pip install aiosqlite"
            ) from exc

        # Strip the "sqlite:///" scheme prefix if present
        path = self._db_url
        for prefix in ("sqlite:///", "sqlite://"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break

        self._conn = await aiosqlite.connect(path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(_CREATE_TASK_OUTCOMES)
        await self._conn.execute(_CREATE_TOOL_REPUTATION)
        await self._conn.execute(_CREATE_GOSSIP_EVENTS)
        await self._conn.commit()

    async def _sqlite_exec(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> None:
        await self._conn.execute(sql, params)
        await self._conn.commit()

    async def _sqlite_fetchall(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[Any]:
        async with self._conn.execute(sql, params) as cursor:
            return await cursor.fetchall()

    async def _sqlite_fetchone(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Any | None:
        async with self._conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    # ------------------------------------------------------------------
    # Private — PostgreSQL backend
    # ------------------------------------------------------------------

    async def _connect_postgres(self) -> None:
        try:
            import asyncpg  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgreSQL persistence. "
                "Install it with: pip install asyncpg"
            ) from exc

        self._conn = await asyncpg.connect(self._db_url)
        await self._conn.execute(_CREATE_TASK_OUTCOMES_PG)
        await self._conn.execute(_CREATE_TOOL_REPUTATION)
        await self._conn.execute(_CREATE_GOSSIP_EVENTS_PG)

    async def _pg_exec(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> None:
        # asyncpg uses $1/$2/... placeholders; convert from ?
        pg_sql = _sqlite_to_pg(sql)
        await self._conn.execute(pg_sql, *params)

    async def _pg_fetch(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[Any]:
        pg_sql = _sqlite_to_pg(sql)
        return await self._conn.fetch(pg_sql, *params)

    async def _pg_fetchrow(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Any | None:
        pg_sql = _sqlite_to_pg(sql)
        return await self._conn.fetchrow(pg_sql, *params)

    # ------------------------------------------------------------------
    # Internal dispatch helpers
    # ------------------------------------------------------------------

    async def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        if self._is_postgres:
            await self._pg_exec(sql, params)
        else:
            await self._sqlite_exec(sql, params)

    async def _fetchall(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[Any]:
        if self._is_postgres:
            return await self._pg_fetch(sql, params)
        return await self._sqlite_fetchall(sql, params)

    async def _fetchone(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> Any | None:
        if self._is_postgres:
            return await self._pg_fetchrow(sql, params)
        return await self._sqlite_fetchone(sql, params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_task_outcome(
        self,
        task: Any,
        action: str,
        outcome: dict[str, Any],
        *,
        task_hash: str | None = None,
        domain: str | None = None,
        stakes: float | None = None,
    ) -> None:
        """Persist a completed task with its action and measured outcome.

        Parameters
        ----------
        task:
            A :class:`~manifold.brain.BrainTask` instance **or** ``None``
            when called with keyword-only overrides (*task_hash*, *domain*,
            *stakes*).
        action:
            The action that was taken (e.g. ``"escalate"``, ``"answer"``).
        outcome:
            Measured outcome dict (arbitrary JSON-serialisable structure).
        task_hash:
            Override prompt hash.  Computed automatically from ``task.prompt``
            when not provided.
        domain:
            Override domain string.  Read from ``task.domain`` when not
            provided.
        stakes:
            Override stakes value.  Read from ``task.stakes`` when not
            provided.
        """
        # Accept both BrainTask and keyword-only overrides for testing
        if task is not None:
            prompt = getattr(task, "prompt", "")
            _domain = domain if domain is not None else getattr(task, "domain", "general")
            _stakes = stakes if stakes is not None else getattr(task, "stakes", 0.5)
        else:
            prompt = ""
            _domain = domain or "general"
            _stakes = stakes or 0.5

        if task_hash is None:
            task_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        _cost = float(outcome.get("cost", 0.0)) if outcome else 0.0
        _risk = float(outcome.get("risk", 0.0)) if outcome else 0.0
        _asset = float(outcome.get("asset", 0.0)) if outcome else 0.0
        _outcome_json = json.dumps(outcome, default=str) if outcome else "{}"
        _now = time.time()

        await self._exec(
            """
            INSERT INTO task_outcomes
                (prompt_hash, domain, stakes, action, cost, risk, asset, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_hash, _domain, _stakes, action, _cost, _risk, _asset, _outcome_json, _now),
        )

    async def get_domain_stats(self, domain: str) -> dict[str, Any]:
        """Return aggregated cost/risk/asset stats for a domain.

        Parameters
        ----------
        domain:
            Domain name (e.g. ``"billing"``, ``"finance"``).

        Returns
        -------
        dict
            Keys: ``total_tasks``, ``escalation_rate``, ``refusal_rate``,
            ``avg_cost``, ``avg_risk``, ``avg_asset``.
        """
        row = await self._fetchone(
            """
            SELECT
                COUNT(*)                                             AS total_tasks,
                SUM(CASE WHEN action = 'escalate' THEN 1 ELSE 0 END) AS escalations,
                SUM(CASE WHEN action = 'refuse'   THEN 1 ELSE 0 END) AS refusals,
                AVG(cost)                                            AS avg_cost,
                AVG(risk)                                            AS avg_risk,
                AVG(asset)                                           AS avg_asset
            FROM task_outcomes
            WHERE domain = ?
            """,
            (domain,),
        )

        if row is None:
            return {
                "domain": domain,
                "total_tasks": 0,
                "escalation_rate": 0.0,
                "refusal_rate": 0.0,
                "avg_cost": 0.0,
                "avg_risk": 0.0,
                "avg_asset": 0.0,
            }

        total = int(row[0]) if row[0] else 0
        escalations = int(row[1]) if row[1] else 0
        refusals = int(row[2]) if row[2] else 0
        avg_cost = float(row[3]) if row[3] else 0.0
        avg_risk = float(row[4]) if row[4] else 0.0
        avg_asset = float(row[5]) if row[5] else 0.0

        return {
            "domain": domain,
            "total_tasks": total,
            "escalation_rate": escalations / total if total else 0.0,
            "refusal_rate": refusals / total if total else 0.0,
            "avg_cost": avg_cost,
            "avg_risk": avg_risk,
            "avg_asset": avg_asset,
        }

    async def save_tool_reputation(
        self,
        tool_name: str,
        score: float,
        timestamp: float | None = None,
    ) -> None:
        """Log a reputation update for a tool.

        Parameters
        ----------
        tool_name:
            Tool identifier (e.g. ``"gpt-4o"``).
        score:
            Reliability score in [0, 1].
        timestamp:
            POSIX timestamp.  Defaults to ``time.time()``.
        """
        _ts = timestamp if timestamp is not None else time.time()
        try:
            await self._exec(
                "INSERT INTO tool_reputation (tool_name, score, recorded_at) VALUES (?, ?, ?)",
                (tool_name, float(score), _ts),
            )
        except Exception as exc:
            # Silently ignore duplicate PRIMARY KEY conflicts (same tool/timestamp).
            # Re-raise all other errors (connection issues, type errors, etc.).
            exc_str = str(exc).lower()
            if "unique" in exc_str or "primary key" in exc_str or "duplicate" in exc_str:
                return
            raise

    async def get_tool_history(
        self,
        tool_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent reputation history for a tool.

        Parameters
        ----------
        tool_name:
            Tool identifier.
        limit:
            Maximum number of records to return (most-recent first).

        Returns
        -------
        list[dict]
            Each dict has ``tool_name``, ``score``, ``recorded_at`` keys.
        """
        rows = await self._fetchall(
            """
            SELECT tool_name, score, recorded_at
            FROM tool_reputation
            WHERE tool_name = ?
            ORDER BY recorded_at DESC
            LIMIT ?
            """,
            (tool_name, limit),
        )
        return [
            {
                "tool_name": str(row[0]),
                "score": float(row[1]),
                "recorded_at": float(row[2]),
            }
            for row in rows
        ]

    async def save_gossip_event(
        self,
        source_id: str,
        target_id: str,
        risk_vector: list[float],
    ) -> None:
        """Record a gossip propagation event.

        Parameters
        ----------
        source_id:
            Originating node / org identifier.
        target_id:
            Receiving node / org identifier.
        risk_vector:
            4-element list ``[cost, risk, neutrality, asset]``.
        """
        _now = time.time()
        _vec_json = json.dumps([float(v) for v in risk_vector])
        await self._exec(
            """
            INSERT INTO gossip_events (source_id, target_id, risk_vector, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, _vec_json, _now),
        )

    # ------------------------------------------------------------------
    # Counters (used by the metrics endpoint)
    # ------------------------------------------------------------------

    async def task_count(self) -> int:
        """Return total number of persisted task outcomes."""
        row = await self._fetchone("SELECT COUNT(*) FROM task_outcomes")
        return int(row[0]) if row and row[0] else 0

    async def escalation_count(self) -> int:
        """Return total number of escalated tasks."""
        row = await self._fetchone(
            "SELECT COUNT(*) FROM task_outcomes WHERE action = 'escalate'"
        )
        return int(row[0]) if row and row[0] else 0

    async def refusal_count(self) -> int:
        """Return total number of refused tasks."""
        row = await self._fetchone(
            "SELECT COUNT(*) FROM task_outcomes WHERE action = 'refuse'"
        )
        return int(row[0]) if row and row[0] else 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sqlite_to_pg(sql: str) -> str:
    """Convert SQLite ``?`` placeholders to asyncpg ``$N`` placeholders."""
    parts = sql.split("?")
    result = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        result += f"${i}" + part
    return result
