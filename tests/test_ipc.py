"""Tests for Phase 49: IPC Event Bus (manifold/ipc.py)."""

from __future__ import annotations

import queue
import threading
import time

import pytest

from manifold.ipc import (
    TOPIC_ADMIN_VETO,
    TOPIC_SANDBOX_VIOLATION,
    Event,
    EventBus,
    _topic_matches,
)


# ---------------------------------------------------------------------------
# _topic_matches helper
# ---------------------------------------------------------------------------


class TestTopicMatches:
    def test_exact_match(self) -> None:
        assert _topic_matches("sandbox.violation", "sandbox.violation") is True

    def test_exact_no_match(self) -> None:
        assert _topic_matches("sandbox.timeout", "sandbox.violation") is False

    def test_wildcard_star_matches_everything(self) -> None:
        assert _topic_matches("anything.here.at.all", "*") is True

    def test_prefix_wildcard_matches_sub_topic(self) -> None:
        assert _topic_matches("sandbox.violation", "sandbox.*") is True

    def test_prefix_wildcard_matches_exact_prefix(self) -> None:
        assert _topic_matches("sandbox", "sandbox.*") is True

    def test_prefix_wildcard_no_match_different_prefix(self) -> None:
        assert _topic_matches("dht.peer.joined", "sandbox.*") is False

    def test_prefix_wildcard_deep_subtopic(self) -> None:
        assert _topic_matches("system.entropy.high.critical", "system.*") is True

    def test_empty_pattern_matches_only_empty_topic(self) -> None:
        assert _topic_matches("", "") is True
        assert _topic_matches("sandbox", "") is False


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class TestEvent:
    def test_frozen(self) -> None:
        e = Event(topic="test")
        with pytest.raises((AttributeError, TypeError)):
            e.topic = "other"  # type: ignore[misc]

    def test_default_payload(self) -> None:
        e = Event(topic="test")
        assert e.payload == {}

    def test_timestamp_auto_set(self) -> None:
        before = time.time()
        e = Event(topic="test")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_to_dict(self) -> None:
        e = Event(topic="sandbox.violation", payload={"hash": "abc"}, timestamp=1.0)
        d = e.to_dict()
        assert d["topic"] == "sandbox.violation"
        assert d["payload"] == {"hash": "abc"}
        assert d["timestamp"] == 1.0

    def test_to_dict_payload_is_copy(self) -> None:
        payload = {"key": "val"}
        e = Event(topic="t", payload=payload)
        d = e.to_dict()
        d["payload"]["key"] = "changed"
        assert e.payload["key"] == "val"


# ---------------------------------------------------------------------------
# EventBus — subscribe / unsubscribe
# ---------------------------------------------------------------------------


class TestEventBusSubscription:
    def test_subscribe_returns_queue(self) -> None:
        bus = EventBus()
        q = bus.subscribe("test.topic")
        assert isinstance(q, queue.Queue)

    def test_subscriber_count_increases(self) -> None:
        bus = EventBus()
        bus.subscribe("test.topic")
        bus.subscribe("test.topic")
        assert bus.subscriber_count("test.topic") == 2

    def test_subscriber_total_count(self) -> None:
        bus = EventBus()
        bus.subscribe("a")
        bus.subscribe("b")
        assert bus.subscriber_count() == 2

    def test_unsubscribe_removes_queue(self) -> None:
        bus = EventBus()
        q = bus.subscribe("test.topic")
        bus.unsubscribe("test.topic", q)
        assert bus.subscriber_count("test.topic") == 0

    def test_unsubscribe_nonexistent_does_not_raise(self) -> None:
        bus = EventBus()
        q: queue.Queue[Event] = queue.Queue()
        bus.unsubscribe("ghost.topic", q)  # should not raise

    def test_registered_patterns(self) -> None:
        bus = EventBus()
        bus.subscribe("a.b")
        bus.subscribe("c.d")
        patterns = bus.registered_patterns()
        assert "a.b" in patterns
        assert "c.d" in patterns


# ---------------------------------------------------------------------------
# EventBus — publish (exact match)
# ---------------------------------------------------------------------------


class TestEventBusPublishExact:
    def test_exact_subscriber_receives_event(self) -> None:
        bus = EventBus()
        q = bus.subscribe(TOPIC_SANDBOX_VIOLATION)
        bus.publish(TOPIC_SANDBOX_VIOLATION, {"hash": "abc"})
        event = q.get_nowait()
        assert event.topic == TOPIC_SANDBOX_VIOLATION
        assert event.payload["hash"] == "abc"

    def test_publish_returns_delivery_count(self) -> None:
        bus = EventBus()
        bus.subscribe("t")
        bus.subscribe("t")
        delivered = bus.publish("t")
        assert delivered == 2

    def test_non_matching_subscriber_does_not_receive(self) -> None:
        bus = EventBus()
        q = bus.subscribe("other.topic")
        bus.publish(TOPIC_SANDBOX_VIOLATION)
        assert q.empty()

    def test_publish_no_subscribers_returns_zero(self) -> None:
        bus = EventBus()
        assert bus.publish("nowhere") == 0

    def test_multiple_publishers_multiple_subscribers(self) -> None:
        bus = EventBus()
        q1 = bus.subscribe("t")
        q2 = bus.subscribe("t")
        bus.publish("t", {"n": 1})
        bus.publish("t", {"n": 2})
        events1 = [q1.get_nowait(), q1.get_nowait()]
        events2 = [q2.get_nowait(), q2.get_nowait()]
        assert [e.payload["n"] for e in events1] == [1, 2]
        assert [e.payload["n"] for e in events2] == [1, 2]


# ---------------------------------------------------------------------------
# EventBus — publish (wildcard)
# ---------------------------------------------------------------------------


class TestEventBusPublishWildcard:
    def test_wildcard_subscriber_receives_sub_topics(self) -> None:
        bus = EventBus()
        q = bus.subscribe("sandbox.*")
        bus.publish("sandbox.violation")
        bus.publish("sandbox.timeout")
        e1 = q.get_nowait()
        e2 = q.get_nowait()
        assert {e1.topic, e2.topic} == {"sandbox.violation", "sandbox.timeout"}

    def test_star_subscriber_receives_all(self) -> None:
        bus = EventBus()
        q = bus.subscribe("*")
        bus.publish("sandbox.violation")
        bus.publish("dht.peer.joined")
        bus.publish(TOPIC_ADMIN_VETO)
        received = []
        for _ in range(3):
            received.append(q.get_nowait())
        assert len(received) == 3

    def test_wildcard_does_not_receive_unrelated_topic(self) -> None:
        bus = EventBus()
        q = bus.subscribe("sandbox.*")
        bus.publish("dht.peer.joined")
        assert q.empty()


# ---------------------------------------------------------------------------
# EventBus — queue full (drop behaviour)
# ---------------------------------------------------------------------------


class TestEventBusQueueFull:
    def test_full_queue_does_not_raise(self) -> None:
        bus = EventBus(max_queue_size=2)
        q = bus.subscribe("t")
        # Overflow the queue
        for _ in range(5):
            bus.publish("t")
        # queue should have exactly max_queue_size items
        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == 2

    def test_delivery_count_excludes_dropped(self) -> None:
        bus = EventBus(max_queue_size=1)
        q = bus.subscribe("t")
        bus.publish("t")  # delivered
        delivered = bus.publish("t")  # dropped (queue full)
        assert delivered == 0
        q.get_nowait()  # drain


# ---------------------------------------------------------------------------
# EventBus — thread safety
# ---------------------------------------------------------------------------


class TestEventBusThreadSafety:
    def test_concurrent_publish_and_subscribe(self) -> None:
        bus = EventBus()
        results: list[Event] = []
        lock = threading.Lock()

        def publisher() -> None:
            for i in range(20):
                bus.publish("stress.test", {"i": i})

        q = bus.subscribe("stress.test")

        def consumer() -> None:
            for _ in range(20):
                try:
                    e = q.get(timeout=2.0)
                    with lock:
                        results.append(e)
                except queue.Empty:
                    break

        t_pub = threading.Thread(target=publisher)
        t_con = threading.Thread(target=consumer)
        t_con.start()
        t_pub.start()
        t_pub.join(timeout=5)
        t_con.join(timeout=5)
        assert len(results) == 20

    def test_multiple_publishers_no_deadlock(self) -> None:
        bus = EventBus()
        errors: list[Exception] = []

        def pub_worker(n: int) -> None:
            try:
                for i in range(10):
                    bus.publish(f"t{n}", {"i": i})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=pub_worker, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors


# ---------------------------------------------------------------------------
# EventBus — wait_for_event
# ---------------------------------------------------------------------------


class TestEventBusWaitForEvent:
    def test_wait_receives_event(self) -> None:
        bus = EventBus()
        q = bus.subscribe("t")

        def delayed_publish() -> None:
            time.sleep(0.05)
            bus.publish("t", {"val": 42})

        threading.Thread(target=delayed_publish, daemon=True).start()
        event = bus.wait_for_event(q, timeout=2.0)
        assert event is not None
        assert event.payload["val"] == 42

    def test_wait_returns_none_on_timeout(self) -> None:
        bus = EventBus()
        q = bus.subscribe("t")
        result = bus.wait_for_event(q, timeout=0.05)
        assert result is None


# ---------------------------------------------------------------------------
# Standard topic constants
# ---------------------------------------------------------------------------


class TestTopicConstants:
    def test_constants_are_strings(self) -> None:
        from manifold.ipc import (
            TOPIC_ADMIN_VETO,
            TOPIC_DHT_PEER_DROPPED,
            TOPIC_DHT_PEER_JOINED,
            TOPIC_META_CHAMPION_PROMOTED,
            TOPIC_SANDBOX_TIMEOUT,
            TOPIC_SANDBOX_VIOLATION,
            TOPIC_SYSTEM_ENTROPY_HIGH,
            TOPIC_VECTOR_ENTRY_ADDED,
        )

        constants = [
            TOPIC_ADMIN_VETO,
            TOPIC_DHT_PEER_DROPPED,
            TOPIC_DHT_PEER_JOINED,
            TOPIC_META_CHAMPION_PROMOTED,
            TOPIC_SANDBOX_TIMEOUT,
            TOPIC_SANDBOX_VIOLATION,
            TOPIC_SYSTEM_ENTROPY_HIGH,
            TOPIC_VECTOR_ENTRY_ADDED,
        ]
        for c in constants:
            assert isinstance(c, str)
            assert len(c) > 0
