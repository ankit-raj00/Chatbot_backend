"""
Tests for voice/config.py's _SarvamKeyPool — the key-rotation logic behind
the multi-key Sarvam fallback in sarvam_tts.py/sarvam_stt.py.

Pure in-memory state, no network — the actual connect-and-retry behavior
(rotating past a rejected key mid-stream) can't be exercised without hitting
the real Sarvam API, so this covers the one piece that's cleanly testable in
isolation: the pool's own bookkeeping.
"""
from voice.config import _SarvamKeyPool


def test_starts_on_the_first_key():
    pool = _SarvamKeyPool(["k1", "k2", "k3"])
    assert pool.current() == "k1"
    assert len(pool) == 3


def test_mark_exhausted_advances_to_next_key():
    pool = _SarvamKeyPool(["k1", "k2", "k3"])
    pool.mark_exhausted("k1")
    assert pool.current() == "k2"
    pool.mark_exhausted("k2")
    assert pool.current() == "k3"


def test_mark_exhausted_on_the_last_key_is_a_noop():
    """No fourth key exists to advance to — must not raise or wrap around,
    since wrapping back to a key already known-bad would just retry it
    forever with zero chance of success."""
    pool = _SarvamKeyPool(["k1", "k2"])
    pool.mark_exhausted("k1")
    pool.mark_exhausted("k2")
    assert pool.current() == "k2"
    pool.mark_exhausted("k2")  # still a no-op, not an error
    assert pool.current() == "k2"


def test_mark_exhausted_is_idempotent_for_a_stale_key():
    """Two concurrent bursts can both report the SAME already-superseded key
    exhausted (a request in flight when the pool already rotated past it) —
    must not double-advance past a key that wasn't actually current."""
    pool = _SarvamKeyPool(["k1", "k2", "k3"])
    pool.mark_exhausted("k1")
    assert pool.current() == "k2"
    pool.mark_exhausted("k1")  # stale report, k1 isn't current anymore
    assert pool.current() == "k2"  # unchanged, not skipped to k3


def test_empty_pool_current_is_blank_and_len_is_zero():
    pool = _SarvamKeyPool([])
    assert pool.current() == ""
    assert len(pool) == 0


def test_single_key_pool_never_advances():
    pool = _SarvamKeyPool(["only-one"])
    pool.mark_exhausted("only-one")
    assert pool.current() == "only-one"


def test_env_parsing_splits_and_strips_comma_separated_keys(monkeypatch):
    """The whole point of the design: SARVAM_API_KEY itself can hold a
    comma-separated list, so no new env var name is needed anywhere in
    deploy.yml/merge_env.sh/test.yml."""
    monkeypatch.setenv("SARVAM_API_KEY", " k1 , k2,k3 ")
    import importlib
    import voice.config as cfg
    importlib.reload(cfg)
    try:
        assert cfg.SARVAM_API_KEYS == ["k1", "k2", "k3"]
        assert cfg.SARVAM_KEY_POOL.current() == "k1"
    finally:
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        importlib.reload(cfg)
