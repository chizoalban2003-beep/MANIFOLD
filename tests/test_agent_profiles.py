"""Tests for manifold/agent_profiles.py."""
from __future__ import annotations

import pytest

from manifold.agent_profiles import AGENT_PROFILES, get_profile, list_profiles


# ---------------------------------------------------------------------------
# AGENT_PROFILES structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"display_name", "capabilities", "domain", "layer", "level", "crna_profile"}
REQUIRED_CRNA = {"c", "r", "n", "a"}


def test_profiles_not_empty():
    assert len(AGENT_PROFILES) >= 12


def test_all_profiles_have_required_keys():
    for name, prof in AGENT_PROFILES.items():
        missing = REQUIRED_KEYS - prof.keys()
        assert not missing, f"Profile {name!r} missing keys: {missing}"


def test_all_crna_profiles_have_four_values():
    for name, prof in AGENT_PROFILES.items():
        crna = prof["crna_profile"]
        missing = REQUIRED_CRNA - crna.keys()
        assert not missing, f"Profile {name!r} crna_profile missing: {missing}"


def test_all_crna_values_in_range():
    for name, prof in AGENT_PROFILES.items():
        for k, v in prof["crna_profile"].items():
            assert 0.0 <= v <= 1.0, (
                f"Profile {name!r} crna_profile[{k!r}] = {v} out of [0, 1]"
            )


def test_all_capabilities_are_nonempty_lists():
    for name, prof in AGENT_PROFILES.items():
        caps = prof["capabilities"]
        assert isinstance(caps, list) and len(caps) > 0, (
            f"Profile {name!r} capabilities must be a non-empty list"
        )


def test_level_is_integer_1_to_5():
    for name, prof in AGENT_PROFILES.items():
        assert isinstance(prof["level"], int) and 1 <= prof["level"] <= 5, (
            f"Profile {name!r} level={prof['level']} not in 1–5"
        )


# ---------------------------------------------------------------------------
# Known profile spot-checks
# ---------------------------------------------------------------------------

def test_claude_sonnet_profile():
    p = AGENT_PROFILES["claude-sonnet"]
    assert p["level"] == 4
    assert "reasoning" in p["capabilities"]
    assert p["domain"] == "general"
    assert p["layer"] == "digital/llm"


def test_claude_opus_is_level_5():
    assert AGENT_PROFILES["claude-opus"]["level"] == 5


def test_roomba_is_physical():
    p = AGENT_PROFILES["roomba"]
    assert p["domain"] == "physical/floor"
    assert "vacuum" in p["capabilities"]


def test_drone_is_physical_aerial():
    p = AGENT_PROFILES["drone"]
    assert p["domain"] == "physical/aerial"
    assert "aerial_nav" in p["capabilities"]


def test_home_assistant_layer():
    assert AGENT_PROFILES["home-assistant"]["layer"] == "physical/device"


def test_openai_swarm_has_coordination():
    assert "coordination" in AGENT_PROFILES["openai-swarm"]["capabilities"]


# ---------------------------------------------------------------------------
# get_profile()
# ---------------------------------------------------------------------------

def test_get_profile_returns_copy():
    p1 = get_profile("claude-sonnet")
    p2 = get_profile("claude-sonnet")
    p1["level"] = 99
    assert p2["level"] != 99


def test_get_profile_valid_name():
    p = get_profile("gpt-4o")
    assert p["display_name"] == "GPT-4o"
    assert "vision" in p["capabilities"]


def test_get_profile_case_insensitive_strip():
    p = get_profile("  GPT-4O  ")
    assert p["display_name"] == "GPT-4o"


def test_get_profile_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown agent profile"):
        get_profile("does-not-exist")


def test_get_profile_error_lists_available():
    with pytest.raises(ValueError, match="claude-sonnet"):
        get_profile("unknown-xyz")


def test_get_profile_roomba():
    p = get_profile("roomba")
    assert p["domain"] == "physical/floor"


def test_get_profile_langchain():
    p = get_profile("langchain")
    assert "memory" in p["capabilities"]


# ---------------------------------------------------------------------------
# list_profiles()
# ---------------------------------------------------------------------------

def test_list_profiles_sorted():
    names = list_profiles()
    assert names == sorted(names)


def test_list_profiles_contains_all_keys():
    assert set(list_profiles()) == set(AGENT_PROFILES)


def test_list_profiles_includes_drone():
    assert "drone" in list_profiles()
