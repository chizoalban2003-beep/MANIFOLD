"""manifold/agent_profiles.py — Pre-filled agent configurations for common agent types.

Use AGENT_PROFILES to skip manual JSON construction when registering well-known
agent types. Each profile contains display_name, capabilities, domain, layer,
level, and a sensible crna_profile (default C/R/N/A values for this agent type).

Example
-------
>>> from manifold.agent_profiles import get_profile
>>> profile = get_profile("claude-sonnet")
>>> profile["capabilities"]
['reasoning', 'summarise', 'code', 'analysis']
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

AGENT_PROFILES: dict[str, dict] = {
    "claude-sonnet": {
        "display_name": "Claude Sonnet",
        "capabilities": ["reasoning", "summarise", "code", "analysis"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 4,
        "crna_profile": {"c": 0.25, "r": 0.20, "n": 0.30, "a": 0.80},
    },
    "claude-opus": {
        "display_name": "Claude Opus",
        "capabilities": ["reasoning", "research", "writing", "code"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 5,
        "crna_profile": {"c": 0.30, "r": 0.20, "n": 0.20, "a": 0.90},
    },
    "gpt-4o": {
        "display_name": "GPT-4o",
        "capabilities": ["reasoning", "vision", "code", "search"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 4,
        "crna_profile": {"c": 0.25, "r": 0.20, "n": 0.25, "a": 0.85},
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "capabilities": ["reasoning", "summarise", "classify"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 3,
        "crna_profile": {"c": 0.15, "r": 0.15, "n": 0.35, "a": 0.70},
    },
    "gemini-pro": {
        "display_name": "Gemini Pro",
        "capabilities": ["reasoning", "vision", "search", "multimodal"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 4,
        "crna_profile": {"c": 0.25, "r": 0.20, "n": 0.25, "a": 0.85},
    },
    "llama-3": {
        "display_name": "Llama 3",
        "capabilities": ["reasoning", "summarise", "local"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 3,
        "crna_profile": {"c": 0.10, "r": 0.15, "n": 0.40, "a": 0.65},
    },
    "mistral": {
        "display_name": "Mistral",
        "capabilities": ["reasoning", "code", "french"],
        "domain": "general",
        "layer": "digital/llm",
        "level": 3,
        "crna_profile": {"c": 0.10, "r": 0.15, "n": 0.40, "a": 0.65},
    },
    "roomba": {
        "display_name": "Roomba",
        "capabilities": ["vacuum", "map", "floor_nav", "bump_detect"],
        "domain": "physical/floor",
        "layer": "physical/floor",
        "level": 2,
        "crna_profile": {"c": 0.40, "r": 0.50, "n": 0.60, "a": 0.40},
    },
    "drone": {
        "display_name": "Drone",
        "capabilities": ["scout", "map", "aerial_nav", "camera"],
        "domain": "physical/aerial",
        "layer": "physical/aerial",
        "level": 3,
        "crna_profile": {"c": 0.50, "r": 0.65, "n": 0.50, "a": 0.55},
    },
    "home-assistant": {
        "display_name": "Home Assistant",
        "capabilities": ["iot", "mqtt", "automation", "sensors"],
        "domain": "physical/home",
        "layer": "physical/device",
        "level": 2,
        "crna_profile": {"c": 0.20, "r": 0.30, "n": 0.50, "a": 0.50},
    },
    "langchain": {
        "display_name": "LangChain Agent",
        "capabilities": ["search", "memory", "tools", "chains"],
        "domain": "general",
        "layer": "digital/framework",
        "level": 3,
        "crna_profile": {"c": 0.20, "r": 0.25, "n": 0.35, "a": 0.70},
    },
    "openai-swarm": {
        "display_name": "OpenAI Swarm",
        "capabilities": ["coordination", "delegation", "parallel"],
        "domain": "general",
        "layer": "digital/swarm",
        "level": 4,
        "crna_profile": {"c": 0.30, "r": 0.25, "n": 0.30, "a": 0.80},
    },
}


def get_profile(name: str) -> dict:
    """Return a copy of the named agent profile.

    Parameters
    ----------
    name:
        Short profile name (e.g. ``"claude-sonnet"``, ``"roomba"``).

    Returns
    -------
    dict
        A copy of the profile dict containing ``display_name``,
        ``capabilities``, ``domain``, ``layer``, ``level``, and
        ``crna_profile``.

    Raises
    ------
    ValueError
        If *name* is not found.  The error message lists all available names.
    """
    key = name.strip().lower()
    if key not in AGENT_PROFILES:
        available = ", ".join(sorted(AGENT_PROFILES))
        raise ValueError(
            f"Unknown agent profile {name!r}. "
            f"Available profiles: {available}"
        )
    return dict(AGENT_PROFILES[key])


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(AGENT_PROFILES)
