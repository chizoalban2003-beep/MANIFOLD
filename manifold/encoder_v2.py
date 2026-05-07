"""Phase 66: Semantic Encoder Upgrade — sentence-embedding-based prompt encoding.

Falls back silently to keyword matching if sentence_transformers is not installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_ENCODER_MODEL = None
_ENCODER_BACKEND: str = "keyword"

def _try_load_model() -> None:
    global _ENCODER_MODEL, _ENCODER_BACKEND
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _ENCODER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _ENCODER_BACKEND = "semantic"
    except Exception:
        _ENCODER_BACKEND = "keyword"

_try_load_model()

# ---------------------------------------------------------------------------
# EncodedTask
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodedTask:
    cost: float = 0.0
    risk: float = 0.0
    neutrality: float = 0.0
    asset: float = 0.0

    def as_vector(self) -> tuple[float, float, float, float]:
        return (self.cost, self.risk, self.neutrality, self.asset)

# ---------------------------------------------------------------------------
# Keyword fallback implementation
# ---------------------------------------------------------------------------

_RISK_WORDS = {"delete", "remove", "payment", "transfer", "override", "bypass", "critical", "password"}
_COST_WORDS = {"verify", "check", "confirm", "retrieve", "fetch", "audit"}
_ASSET_WORDS = {"resolve", "fix", "help", "complete", "deliver", "generate"}

def _keyword_encode(prompt: str) -> EncodedTask:
    words = set(prompt.lower().split())
    risk = min(1.0, len(words & _RISK_WORDS) * 0.18)
    cost = min(1.0, len(words & _COST_WORDS) * 0.18)
    asset = min(1.0, len(words & _ASSET_WORDS) * 0.18)
    neutrality = max(0.0, 1.0 - risk - cost)
    return EncodedTask(cost=cost, risk=risk, neutrality=neutrality, asset=asset)

# ---------------------------------------------------------------------------
# Semantic implementation
# ---------------------------------------------------------------------------

_RISK_ANCHORS = [
    "delete all data permanently",
    "send money transfer payment refund",
    "override safety check bypass restriction",
    "critical production system failure",
    "unauthorized access credentials password",
]
_COST_ANCHORS = ["slow careful verification", "retrieve large dataset"]
_ASSET_ANCHORS = ["resolve customer complaint", "complete task deliver value"]

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)

def _semantic_encode(prompt: str) -> EncodedTask:
    try:
        embeddings = _ENCODER_MODEL.encode([prompt] + _RISK_ANCHORS + _COST_ANCHORS + _ASSET_ANCHORS)
        prompt_emb = embeddings[0].tolist()
        idx = 1
        risk_sims = [_cosine(prompt_emb, embeddings[i].tolist()) for i in range(idx, idx + len(_RISK_ANCHORS))]
        idx += len(_RISK_ANCHORS)
        cost_sims = [_cosine(prompt_emb, embeddings[i].tolist()) for i in range(idx, idx + len(_COST_ANCHORS))]
        idx += len(_COST_ANCHORS)
        asset_sims = [_cosine(prompt_emb, embeddings[i].tolist()) for i in range(idx, idx + len(_ASSET_ANCHORS))]
        risk = min(1.0, max(risk_sims) * 2.0)
        cost = min(1.0, max(cost_sims) * 2.0)
        asset = min(1.0, max(asset_sims) * 2.0)
        neutrality = max(0.0, 1.0 - risk - cost)
        return EncodedTask(cost=cost, risk=risk, neutrality=neutrality, asset=asset)
    except Exception:
        return _keyword_encode(prompt)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_prompt(prompt: str, force_keyword: bool = False) -> EncodedTask:
    """Encode a prompt into an EncodedTask with cost/risk/neutrality/asset scores."""
    if force_keyword or _ENCODER_BACKEND == "keyword":
        return _keyword_encode(prompt)
    return _semantic_encode(prompt)

def encoder_backend() -> str:
    """Return "semantic" if sentence_transformers loaded, else "keyword"."""
    return _ENCODER_BACKEND
