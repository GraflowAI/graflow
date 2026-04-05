"""Tests for MemoryChannel thread safety and race conditions.

Demonstrates that concurrent read-modify-write on MemoryChannel
without locks leads to lost updates (race condition), and that
the ``atomic_add()`` / ``lock()`` APIs solve the problem.

CPython's GIL makes individual dict operations (get/set) effectively atomic,
but the *compound* read-modify-write pattern is NOT atomic. The race window
is between get() and set(). We insert time.sleep(0) to yield the GIL and
force context switches, reliably reproducing the lost-update problem.
"""

from __future__ import annotations

import threading
import time

import pytest

from graflow.channels.memory_channel import MemoryChannel


class TestRaceConditionReproduction:
    """Prove that naive get/set is unsafe under concurrency."""

    def test_concurrent_get_set_loses_updates(self) -> None:
        """Naive get → sleep(0) → set loses updates due to interleaving."""
        channel = MemoryChannel("test")
        channel.set("counter", 0)

        num_threads = 10
        increments_per_thread = 100
        expected = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def unsafe_increment() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                val = channel.get("counter")
                time.sleep(0)  # Yield GIL → forces interleaving
                channel.set("counter", val + 1)

        threads = [threading.Thread(target=unsafe_increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = channel.get("counter")
        assert actual < expected, (
            f"Expected lost updates but got exact count {actual}. "
            f"Extremely unlikely with {num_threads} threads and sleep(0)."
        )


class TestAtomicAdd:
    """Tests for the atomic ``atomic_add()`` method."""

    def test_concurrent_add_no_lost_updates(self) -> None:
        """Concurrent atomic_add() must not lose any updates."""
        channel = MemoryChannel("test")
        channel.set("counter", 0)

        num_threads = 10
        increments_per_thread = 1000
        expected = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def atomic_add() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                channel.atomic_add("counter")

        threads = [threading.Thread(target=atomic_add) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = channel.get("counter")
        assert actual == expected, f"Lost updates: {actual} != {expected}"

    def test_add_positive(self) -> None:
        channel = MemoryChannel("test")
        channel.set("counter", 10)
        result = channel.atomic_add("counter", 5)
        assert result == 15
        assert channel.get("counter") == 15

    def test_add_negative(self) -> None:
        """Negative amount works as decrement."""
        channel = MemoryChannel("test")
        channel.set("counter", 10)
        result = channel.atomic_add("counter", -3)
        assert result == 7

    def test_add_initializes_missing_key(self) -> None:
        """Missing key is initialised to 0 before adding."""
        channel = MemoryChannel("test")
        result = channel.atomic_add("new_key")
        assert result == 1
        assert channel.get("new_key") == 1

    def test_add_float(self) -> None:
        channel = MemoryChannel("test")
        channel.set("metric", 1.5)
        result = channel.atomic_add("metric", 0.25)
        assert result == pytest.approx(1.75)

    def test_add_raises_on_non_numeric(self) -> None:
        channel = MemoryChannel("test")
        channel.set("data", "hello")
        with pytest.raises(TypeError, match="expected int or float"):
            channel.atomic_add("data", 1)

    def test_set_and_atomic_add_share_same_key(self) -> None:
        """set() then atomic_add() then get() must all operate on the same value."""
        channel = MemoryChannel("test")
        channel.set("counter", 0)
        channel.atomic_add("counter", 5)
        assert channel.get("counter") == 5

        channel.set("counter", 100)
        channel.atomic_add("counter", -10)
        assert channel.get("counter") == 90

    def test_atomic_add_then_set_overwrites(self) -> None:
        """set() after atomic_add() overwrites the value."""
        channel = MemoryChannel("test")
        channel.atomic_add("counter", 42)
        channel.set("counter", 0)
        assert channel.get("counter") == 0

    def test_atomic_add_visible_in_keys_and_exists(self) -> None:
        """Keys created by atomic_add() appear in keys() and exists()."""
        channel = MemoryChannel("test")
        channel.atomic_add("auto_created", 1)
        assert channel.exists("auto_created")
        assert "auto_created" in channel.keys()


class TestAdvisoryLock:
    """Tests for the explicit ``lock()`` context manager."""

    def test_lock_prevents_lost_updates(self) -> None:
        """Compound get-modify-set inside lock() must not lose updates."""
        channel = MemoryChannel("test")
        channel.set("counter", 0)

        num_threads = 10
        increments_per_thread = 100
        expected = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def safe_increment() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                with channel.lock("counter"):
                    val = channel.get("counter")
                    time.sleep(0)  # same yield as the race test
                    channel.set("counter", val + 1)

        threads = [threading.Thread(target=safe_increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = channel.get("counter")
        assert actual == expected, f"Lost updates even with lock: {actual} != {expected}"

    def test_lock_is_reentrant(self) -> None:
        """Same thread can acquire the same lock twice (RLock)."""
        channel = MemoryChannel("test")
        channel.set("x", 0)
        with channel.lock("x"):
            with channel.lock("x"):  # must not deadlock
                channel.atomic_add("x", 1)
        assert channel.get("x") == 1

    def test_lock_timeout_raises(self) -> None:
        """If lock is held by another thread, timeout triggers TimeoutError."""
        channel = MemoryChannel("test")
        held = threading.Event()
        done = threading.Event()

        def holder() -> None:
            with channel.lock("key"):
                held.set()
                done.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        held.wait()

        with pytest.raises(TimeoutError):
            with channel.lock("key", timeout=0.1):
                pass  # pragma: no cover

        done.set()
        t.join()

    def test_lock_different_keys_independent(self) -> None:
        """Locks on different keys do not block each other."""
        channel = MemoryChannel("test")
        order: list[str] = []
        barrier = threading.Barrier(2)

        def lock_a() -> None:
            barrier.wait()
            with channel.lock("a"):
                order.append("a-acquired")
                time.sleep(0.05)

        def lock_b() -> None:
            barrier.wait()
            with channel.lock("b"):
                order.append("b-acquired")
                time.sleep(0.05)

        ta = threading.Thread(target=lock_a)
        tb = threading.Thread(target=lock_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert "a-acquired" in order
        assert "b-acquired" in order

    def test_lock_compound_counter_with_overflow(self) -> None:
        """Concurrent counter with threshold-based reset using lock().

        Each thread increments a counter; when it reaches a threshold the
        counter is reset to 0 and an overflow counter is bumped.
        Without the lock, the conditional reset is racy.
        """
        channel = MemoryChannel("test")
        channel.set("counter", 0)
        channel.set("overflow_count", 0)

        threshold = 10
        num_threads = 5
        increments_per_thread = 100
        total_increments = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                with channel.lock("counter"):
                    val = channel.get("counter")
                    if val >= threshold:
                        channel.set("counter", 0)
                        channel.atomic_add("overflow_count", 1)
                    else:
                        channel.set("counter", val + 1)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counter = channel.get("counter")
        overflows = channel.get("overflow_count")

        # Invariant: every increment either bumped the counter or triggered a reset
        # total_increments = overflows * (threshold + 1) + counter
        assert overflows * (threshold + 1) + counter == total_increments, (
            f"Inconsistent state: overflows={overflows}, counter={counter}, expected total={total_increments}"
        )
        assert 0 <= counter <= threshold

    def test_lock_multi_key_update(self) -> None:
        """Demonstrate lock() protecting a multi-key read-modify-write."""
        channel = MemoryChannel("test")
        channel.set("balance_a", 1000)
        channel.set("balance_b", 1000)

        num_threads = 10
        transfers_per_thread = 50

        barrier = threading.Barrier(num_threads)

        def transfer() -> None:
            barrier.wait()
            for _ in range(transfers_per_thread):
                # Lock on a logical "transfer" key to serialize transfers
                with channel.lock("transfer"):
                    a = channel.get("balance_a")
                    b = channel.get("balance_b")
                    time.sleep(0)  # yield to stress-test
                    # Move 1 unit from a to b
                    channel.set("balance_a", a - 1)
                    channel.set("balance_b", b + 1)

        threads = [threading.Thread(target=transfer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        a = channel.get("balance_a")
        b = channel.get("balance_b")
        total_transfers = num_threads * transfers_per_thread

        # Conservation: total should always be 2000
        assert a + b == 2000, f"Balance inconsistency: a={a}, b={b}, sum={a + b}"
        assert a == 1000 - total_transfers
        assert b == 1000 + total_transfers


class TestBaseChannelLockNoop:
    """Verify that the base Channel.lock() default is a no-op."""

    def test_noop_lock(self) -> None:
        """Any subclass that doesn't override lock() gets a no-op."""
        from graflow.channels.base import Channel

        # Create a minimal concrete subclass that does NOT override lock()
        class MinimalChannel(Channel):
            def set(self, key, value, ttl=None):  # type: ignore[override]
                pass

            def get(self, key, default=None):  # type: ignore[override]
                pass

            def delete(self, key):  # type: ignore[override]
                return False

            def exists(self, key):  # type: ignore[override]
                return False

            def keys(self):  # type: ignore[override]
                return []

            def clear(self):
                pass

            def append(self, key, value, ttl=None):  # type: ignore[override]
                return 0

            def prepend(self, key, value, ttl=None):  # type: ignore[override]
                return 0

            def atomic_add(self, key, amount=1):  # type: ignore[override]
                return 0

        ch = MinimalChannel("noop")
        # Should not raise, just pass through
        with ch.lock("anything"):
            pass


class TestSerialization:
    """Verify MemoryChannel survives pickle round-trip (checkpoint/resume)."""

    def test_pickle_round_trip_preserves_data(self) -> None:
        """Data and TTL survive pickle; locks are recreated."""
        import pickle

        channel = MemoryChannel("test")
        channel.set("counter", 42)
        channel.set("config", {"batch": 100})

        # Force lock creation so it exists in __dict__
        channel.atomic_add("counter", 1)

        restored: MemoryChannel = pickle.loads(pickle.dumps(channel))

        assert restored.get("counter") == 43
        assert restored.get("config") == {"batch": 100}
        assert restored.name == "test"

    def test_pickle_round_trip_locks_functional(self) -> None:
        """Locks work correctly after deserialization."""
        import pickle

        channel = MemoryChannel("test")
        channel.set("x", 0)

        restored: MemoryChannel = pickle.loads(pickle.dumps(channel))

        # atomic_add must work (uses per-key lock internally)
        restored.atomic_add("x", 5)
        assert restored.get("x") == 5

        # Advisory lock must work
        with restored.lock("x"):
            val = restored.get("x")
            restored.set("x", val + 1)
        assert restored.get("x") == 6

    def test_cloudpickle_round_trip(self) -> None:
        """Checkpoint uses cloudpickle — verify it works too."""
        import cloudpickle

        channel = MemoryChannel("test")
        channel.set("data", [1, 2, 3])
        channel.atomic_add("counter", 10)

        restored: MemoryChannel = cloudpickle.loads(cloudpickle.dumps(channel))

        assert restored.get("data") == [1, 2, 3]
        assert restored.get("counter") == 10
        restored.atomic_add("counter", 1)
        assert restored.get("counter") == 11

    def test_pickle_round_trip_concurrent_after_restore(self) -> None:
        """Restored channel handles concurrent access correctly."""
        import pickle

        channel = MemoryChannel("test")
        channel.set("counter", 0)

        restored: MemoryChannel = pickle.loads(pickle.dumps(channel))

        num_threads = 5
        increments_per_thread = 200
        expected = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                restored.atomic_add("counter")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert restored.get("counter") == expected
