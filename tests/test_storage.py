import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from voice_summary_bot.storage import QueueFull, Store


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock):
    instance = Store(tmp_path / "data", ttl_hours=1, clock=clock)
    yield instance
    instance.close()


def enqueue(store: Store, unique: str = "audio-one", user: int = 123, **overrides):
    fields = {
        "user_id": user,
        "chat_id": user,
        "message_id": 10,
        "file_id": "download-one",
        "file_unique_id": unique,
        "status_message_id": 11,
    }
    fields.update(overrides)
    return store.enqueue(**fields)


def test_fifo_claims_and_counts_include_processing(store: Store):
    first, created = enqueue(store)
    second, _ = enqueue(store, "audio-two")
    assert created
    assert len(first.id) == 32
    assert store.pending_count() == 2
    assert store.next_job().id == first.id
    assert store.get(first.id, 123, 123).state == "processing"
    assert store.pending_count() == 2
    assert store.next_job().id == second.id
    assert store.next_job() is None


def test_completed_result_is_only_retrievable_by_original_owner_and_chat(store: Store):
    job, _ = enqueue(store)
    store.next_job()
    store.complete(job.id, "source text", "ru", '{"summary":"short"}')
    result = store.get(job.id, 123, 123)
    assert result.state == "done"
    assert result.transcript == "source text"
    assert result.language == "ru"
    assert result.summary_json == '{"summary":"short"}'
    assert store.pending_count() == 0
    assert store.get(job.id, 999, 123) is None
    assert store.get(job.id, 123, 999) is None
    assert store.get("missing", 123, 123) is None


def test_duplicate_does_not_replace_pending_or_done_job(store: Store):
    job, _ = enqueue(store)
    duplicate, created = enqueue(store, message_id=20, status_message_id=21, max_pending=1)
    assert not created
    assert duplicate == job
    store.next_job()
    duplicate, created = enqueue(store)
    assert duplicate.state == "processing"
    assert not created
    store.complete(job.id, "text", "en", "{}")
    duplicate, created = enqueue(store)
    assert duplicate.id == job.id
    assert duplicate.state == "done"
    assert not created


def test_same_audio_from_another_user_is_a_separate_job(store: Store):
    first, _ = enqueue(store)
    second, created = enqueue(store, user=456)
    assert created
    assert second.id != first.id
    assert store.pending_count() == 2


def test_queue_limit_is_global_and_completed_jobs_release_capacity(store: Store):
    first, _ = enqueue(store, max_pending=1)
    store.next_job()
    with pytest.raises(QueueFull):
        enqueue(store, user=456, max_pending=1)
    store.complete(first.id, "text", "en", "{}")
    enqueue(store, user=456, max_pending=1)
    assert store.pending_count() == 1


def test_failed_duplicate_is_requeued_with_fresh_source_ids(store: Store, clock: Clock):
    original, _ = enqueue(store)
    store.next_job()
    store.fail(original.id)
    clock.now += 10
    retried, created = enqueue(
        store, chat_id=456, message_id=20, file_id="new-download", status_message_id=21
    )
    assert not created
    assert retried.id == original.id
    assert retried.state == "queued"
    assert retried.created_at == clock.now
    assert (retried.chat_id, retried.message_id, retried.status_message_id) == (456, 20, 21)
    assert retried.file_id == "new-download"
    assert retried.transcript is retried.summary_json is retried.language is None
    assert store.get(original.id, 123, 123) is None
    assert store.next_job().id == original.id


def test_failed_retry_obeys_queue_cap_and_rolls_back(store: Store):
    failed, _ = enqueue(store)
    store.fail(failed.id)
    enqueue(store, "busy", max_pending=1)
    with pytest.raises(QueueFull):
        enqueue(store, max_pending=1)
    assert store.get(failed.id, 123, 123).state == "failed"
    assert store.pending_count() == 1


def test_summary_failure_keeps_transcript_until_retry_or_expiry(store: Store):
    job, _ = enqueue(store)
    store.next_job()
    store.complete(job.id, "retained transcription", "ru", "")
    store.fail(job.id)
    failed = store.get(job.id, 123, 123)
    assert failed.state == "failed"
    assert failed.transcript == "retained transcription"
    assert failed.language == "ru"
    assert failed.summary_json == ""
    assert "retained transcription" not in repr(failed)
    retried, created = enqueue(store)
    assert retried.transcript is retried.summary_json is retried.language is None
    assert not created


def test_recovery_preserves_queue_and_preferences(tmp_path: Path, clock: Clock):
    directory = tmp_path / "data"
    original = Store(directory, clock=clock)
    first, _ = enqueue(original)
    second, _ = enqueue(original, "second")
    original.next_job()
    original.set_language(123, "ru")
    original.close()
    restarted = Store(directory, clock=clock)
    try:
        assert restarted.get(first.id, 123, 123).state == "processing"
        restarted.recover()
        assert restarted.get_language(123) == "ru"
        assert restarted.next_job().id == first.id
        assert restarted.next_job().id == second.id
    finally:
        restarted.close()


def test_restart_recovers_partial_transcript_for_summary(tmp_path: Path, clock: Clock):
    directory = tmp_path / "data"
    original = Store(directory, clock=clock)
    job, _ = enqueue(original)
    original.next_job()
    original.save_transcript(job.id, "transcribed before restart", "ru")
    assert original.get(job.id, 123, 123).state == "processing"
    original.close()
    restarted = Store(directory, clock=clock)
    try:
        restarted.recover()
        recovered = restarted.next_job()
        assert recovered.id == job.id
        assert recovered.state == "processing"
        assert recovered.transcript == "transcribed before restart"
        assert recovered.language == "ru"
        assert recovered.summary_json is None
        restarted.fail(job.id)
        assert restarted.get(job.id, 123, 123).transcript == "transcribed before restart"
    finally:
        restarted.close()


def test_terminal_ttl_starts_at_finish_and_preserves_unfinished_jobs(store: Store, clock: Clock):
    done, _ = enqueue(store)
    store.next_job()
    clock.now += 4000
    store.complete(done.id, "private-source-marker", "en", "private-summary-marker")
    failed, _ = enqueue(store, "failed")
    store.fail(failed.id)
    queued, _ = enqueue(store, "queued")
    clock.now += 3599
    assert store.get(done.id, 123, 123) is not None
    clock.now += 1
    store.purge()
    assert store.get(done.id, 123, 123) is None
    assert store.get(failed.id, 123, 123) is None
    assert store.get(queued.id, 123, 123).state == "queued"
    for path in store._database_path.parent.glob("queue.sqlite3*"):
        data = path.read_bytes()
        assert b"private-source-marker" not in data
        assert b"private-summary-marker" not in data


def test_expired_results_are_purged_on_startup(tmp_path: Path, clock: Clock):
    directory = tmp_path / "data"
    original = Store(directory, ttl_hours=1, clock=clock)
    job, _ = enqueue(original)
    original.next_job()
    original.complete(job.id, "text", "en", "{}")
    original.close()
    clock.now += 3601
    restarted = Store(directory, ttl_hours=1, clock=clock)
    try:
        assert restarted.get(job.id, 123, 123) is None
        replacement, created = enqueue(restarted)
        assert created
        assert replacement.id != job.id
    finally:
        restarted.close()


def test_periodic_activity_purges_without_get(store: Store, clock: Clock):
    job, _ = enqueue(store)
    store.fail(job.id)
    clock.now += 3601
    store.pending_count()
    with sqlite3.connect(store._database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_preferences_are_isolated_and_validated(store: Store):
    assert store.get_language(123) == "auto"
    assert store.get_language(123, "en") == "en"
    store.set_language(123, "ru")
    assert store.get_language(123) == "ru"
    assert store.get_language(456) == "auto"
    store.set_language(123, "en")
    assert store.get_language(123) == "en"
    with pytest.raises(ValueError):
        store.set_language(123, "de")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_existing_directory_and_database_permissions_are_private(tmp_path: Path):
    directory = tmp_path / "data"
    directory.mkdir(mode=0o755)
    database = directory / "queue.sqlite3"
    database.touch(mode=0o644)
    store = Store(directory)
    try:
        enqueue(store)
        assert directory.stat().st_mode & 0o777 == 0o700
        for path in directory.glob("queue.sqlite3*"):
            assert path.stat().st_mode & 0o777 == 0o600
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        store.close()


def test_two_connections_cannot_claim_the_same_job(tmp_path: Path):
    directory = tmp_path / "data"
    store = Store(directory)
    try:
        original, _ = enqueue(store)

        def claim():
            contender = Store(directory)
            try:
                return contender.next_job()
            finally:
                contender.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: claim(), range(2)))
        assert [result.id for result in results if result is not None] == [original.id]
    finally:
        store.close()
