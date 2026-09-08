"""Private SQLite job queue. Store instances are used on their creating thread."""

import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


class QueueFull(Exception):
    """The queue has reached its limit of queued and processing jobs."""


@dataclass(frozen=True)
class Job:
    id: str
    user_id: int
    chat_id: int
    message_id: int
    file_id: str
    file_unique_id: str
    status_message_id: int
    state: str
    transcript: str | None = field(repr=False)
    summary_json: str | None = field(repr=False)
    language: str | None
    created_at: float
    duration_limit: int = 300


class Store:
    """Durable FIFO queue with terminal-result expiry and per-user preferences.

    The pending limit applies to the whole queue. Results expire ``ttl_hours``
    after completion or failure; unfinished jobs remain available for recovery.
    Only call ``recover`` when starting the sole processing worker.
    """

    def __init__(
        self,
        data_dir: Path,
        ttl_hours: int = 24,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive.")
        self._clock = clock or time.time
        self._ttl_seconds = ttl_hours * 3600
        self._next_purge = 0.0
        data_dir = Path(data_dir)
        if data_dir.is_symlink():
            raise ValueError("The data directory must not be a symbolic link.")
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        data_dir.chmod(0o700)
        self._database_path = data_dir / "queue.sqlite3"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._database_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        self._connection = sqlite3.connect(self._database_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA secure_delete = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    file_id TEXT NOT NULL,
                    file_unique_id TEXT NOT NULL,
                    status_message_id INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('queued', 'processing', 'done', 'failed')),
                    transcript TEXT,
                    summary_json TEXT,
                    language TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    duration_limit INTEGER NOT NULL DEFAULT 300
                        CHECK (duration_limit BETWEEN 1 AND 600),
                    UNIQUE (user_id, file_unique_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_pending ON jobs (state, created_at);
                CREATE INDEX IF NOT EXISTS jobs_expiry ON jobs (state, updated_at);
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id INTEGER PRIMARY KEY,
                    language TEXT NOT NULL CHECK (language IN ('auto', 'en', 'ru'))
                );
                CREATE TABLE IF NOT EXISTS duration_preferences (
                    user_id INTEGER PRIMARY KEY,
                    duration_limit INTEGER NOT NULL CHECK (duration_limit IN (300, 600))
                );
                """
            )
            # Preserve the original five-minute allowance for jobs already queued
            # before duration preferences were introduced. Lock the schema check
            # and migration together so simultaneous openers cannot add it twice.
            with self._transaction():
                columns = {
                    row["name"] for row in self._connection.execute("PRAGMA table_info(jobs)")
                }
                if "duration_limit" not in columns:
                    self._connection.execute(
                        "ALTER TABLE jobs ADD COLUMN duration_limit INTEGER NOT NULL "
                        "DEFAULT 300 CHECK (duration_limit BETWEEN 1 AND 600)"
                    )
            self._private_files()
            self.purge()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def _private_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self._database_path}{suffix}")
            if path.exists():
                path.chmod(0o600)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _job(row: sqlite3.Row | None) -> Job | None:
        if row is None:
            return None
        return Job(**{name: row[name] for name in Job.__dataclass_fields__})

    def recover(self) -> None:
        self._maybe_purge()
        self._connection.execute(
            "UPDATE jobs SET state = 'queued', updated_at = ? WHERE state = 'processing'",
            (self._clock(),),
        )

    def purge(self) -> None:
        now = self._clock()
        with self._transaction():
            self._connection.execute(
                "DELETE FROM jobs WHERE state IN ('done', 'failed') AND updated_at <= ?",
                (now - self._ttl_seconds,),
            )
        # secure_delete clears freed database pages; truncate the WAL so it does
        # not retain older copies of expired text after an ordinary checkpoint.
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._private_files()
        self._next_purge = now + min(60.0, self._ttl_seconds)

    def _maybe_purge(self) -> None:
        if self._clock() >= self._next_purge:
            self.purge()

    def get_language(self, user_id: int, default: str = "auto") -> str:
        self._maybe_purge()
        row = self._connection.execute(
            "SELECT language FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["language"] if row else default

    def set_language(self, user_id: int, language: str) -> None:
        if language not in {"auto", "en", "ru"}:
            raise ValueError("language must be auto, en or ru.")
        self._maybe_purge()
        self._connection.execute(
            "INSERT INTO preferences (user_id, language) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET language = excluded.language",
            (user_id, language),
        )

    def get_duration_limit(self, user_id: int, default: int = 600) -> int:
        self._maybe_purge()
        row = self._connection.execute(
            "SELECT duration_limit FROM duration_preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["duration_limit"] if row else default

    def set_duration_limit(self, user_id: int, duration_limit: int) -> None:
        if type(duration_limit) is not int or duration_limit not in {300, 600}:
            raise ValueError("duration_limit must be 300 or 600 seconds.")
        self._maybe_purge()
        self._connection.execute(
            "INSERT INTO duration_preferences (user_id, duration_limit) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET duration_limit = excluded.duration_limit",
            (user_id, duration_limit),
        )

    def enqueue(
        self,
        user_id: int,
        chat_id: int,
        message_id: int,
        file_id: str,
        file_unique_id: str,
        status_message_id: int,
        max_pending: int = 10,
        duration_limit: int = 300,
    ) -> tuple[Job, bool]:
        """Return the job and whether a new record was created.

        Failed duplicates are requeued with fresh message IDs and duration limit,
        and return False. Other duplicates retain their original allowance.
        """
        if max_pending < 1:
            raise ValueError("max_pending must be positive.")
        if type(duration_limit) is not int or not 1 <= duration_limit <= 600:
            raise ValueError("duration_limit must be an integer between 1 and 600 seconds.")
        # Also expires a duplicate immediately, even between periodic purges.
        self.purge()
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE user_id = ? AND file_unique_id = ?",
                (user_id, file_unique_id),
            ).fetchone()
            existing = self._job(row)
            if existing is not None and existing.state != "failed":
                return existing, False
            if self._pending_count() >= max_pending:
                raise QueueFull("The processing queue is full. Try again later.")
            now = self._clock()
            if existing is not None:
                self._connection.execute(
                    "UPDATE jobs SET chat_id = ?, message_id = ?, file_id = ?, "
                    "status_message_id = ?, state = 'queued', transcript = NULL, "
                    "summary_json = NULL, language = NULL, created_at = ?, updated_at = ?, "
                    "duration_limit = ? "
                    "WHERE id = ?",
                    (
                        chat_id,
                        message_id,
                        file_id,
                        status_message_id,
                        now,
                        now,
                        duration_limit,
                        existing.id,
                    ),
                )
                job_id = existing.id
            else:
                job_id = uuid.uuid4().hex
                self._connection.execute(
                    "INSERT INTO jobs (id, user_id, chat_id, message_id, file_id, "
                    "file_unique_id, status_message_id, state, created_at, updated_at, "
                    "duration_limit) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                    (
                        job_id,
                        user_id,
                        chat_id,
                        message_id,
                        file_id,
                        file_unique_id,
                        status_message_id,
                        now,
                        now,
                        duration_limit,
                    ),
                )
            created = self._job(
                self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )
            assert created is not None
            return created, existing is None

    def next_job(self) -> Job | None:
        self._maybe_purge()
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                "UPDATE jobs SET state = 'processing', updated_at = ? WHERE id = ?",
                (self._clock(), row["id"]),
            )
            return self._job(
                self._connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            )

    def save_transcript(self, job_id: str, transcript: str, language: str) -> None:
        """Persist transcription while the job remains recoverable as processing."""
        self._maybe_purge()
        self._connection.execute(
            "UPDATE jobs SET transcript = ?, language = ?, updated_at = ? "
            "WHERE id = ? AND state = 'processing'",
            (transcript, language, self._clock(), job_id),
        )

    def complete(self, job_id: str, transcript: str, language: str, summary_json: str) -> None:
        self._maybe_purge()
        self._connection.execute(
            "UPDATE jobs SET state = 'done', transcript = ?, language = ?, summary_json = ?, "
            "updated_at = ? WHERE id = ? AND state = 'processing'",
            (transcript, language, summary_json, self._clock(), job_id),
        )

    def fail(self, job_id: str) -> None:
        self._maybe_purge()
        self._connection.execute(
            "UPDATE jobs SET state = 'failed', updated_at = ? "
            "WHERE id = ? AND state IN ('queued', 'processing', 'done')",
            (self._clock(), job_id),
        )

    def get(self, job_id: str, user_id: int, chat_id: int) -> Job | None:
        self.purge()
        return self._job(
            self._connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND user_id = ? AND chat_id = ?",
                (job_id, user_id, chat_id),
            ).fetchone()
        )

    def _pending_count(self) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'processing')"
        ).fetchone()[0]

    def pending_count(self) -> int:
        self._maybe_purge()
        return self._pending_count()
