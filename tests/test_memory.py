"""
Session memory tests — verifies the in-memory conversation store.
"""
from src.memory.session_memory import SessionMemory


def test_memory_add_and_retrieve():
    mem = SessionMemory()
    mem.add_message("sess_1", "user", "hello")
    mem.add_message("sess_1", "assistant", "hi there")

    history = mem.get_history("sess_1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"


def test_memory_separate_sessions():
    mem = SessionMemory()
    mem.add_message("sess_1", "user", "query 1")
    mem.add_message("sess_2", "user", "query 2")

    assert len(mem.get_history("sess_1")) == 1
    assert len(mem.get_history("sess_2")) == 1
    assert mem.get_history("sess_1")[0]["content"] == "query 1"


def test_memory_empty_session():
    mem = SessionMemory()
    assert mem.get_history("nonexistent") == []
    assert not mem.session_exists("nonexistent")


def test_memory_max_turns():
    mem = SessionMemory(max_turns_per_session=3)
    for i in range(5):
        mem.add_message("sess_1", "user", f"msg {i}")

    history = mem.get_history("sess_1")
    assert len(history) == 3
    assert history[0]["content"] == "msg 2"  # oldest trimmed


def test_memory_clear():
    mem = SessionMemory()
    mem.add_message("sess_1", "user", "hello")
    assert mem.session_exists("sess_1")

    mem.clear_session("sess_1")
    assert not mem.session_exists("sess_1")
    assert mem.get_history("sess_1") == []


def test_memory_returns_copy():
    """get_history should return a copy, not a mutable reference."""
    mem = SessionMemory()
    mem.add_message("sess_1", "user", "hello")

    h1 = mem.get_history("sess_1")
    h1.append({"role": "user", "content": "injected"})

    h2 = mem.get_history("sess_1")
    assert len(h2) == 1  # injection didn't affect the store
