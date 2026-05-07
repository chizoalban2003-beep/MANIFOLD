"""GlobalWorkspace — domain routing via keyword competition."""
from __future__ import annotations

import re


_KEYWORDS: dict[str, list[str]] = {
    "finance": [
        "refund", "invoice", "billing", "payment", "transaction", "revenue",
        "budget", "tax", "finance", "ledger",
    ],
    "legal": [
        "contract", "compliance", "gdpr", "clause", "liability", "litigation",
        "attorney", "regulation", "statute", "jurisdiction",
    ],
    "medical": [
        "patient", "diagnosis", "drug", "treatment", "clinical", "symptom",
        "prescription", "surgery", "dosage", "therapy",
    ],
    "devops": [
        "deploy", "pipeline", "container", "kubernetes", "ci", "cd",
        "infrastructure", "monitoring", "kubernetes", "release",
    ],
    "creative": [
        "story", "poem", "creative", "design", "art", "write", "narrative",
        "character", "plot", "illustration",
    ],
    "support": [
        "ticket", "support", "help", "issue", "customer", "escalate",
        "resolution", "agent", "helpdesk", "complaint",
    ],
    "research": [
        "research", "study", "hypothesis", "experiment", "data", "analysis",
        "paper", "academic", "journal", "findings",
    ],
}


class GlobalWorkspace:
    """Routes prompts to the best-matching domain via keyword competition."""

    def __init__(self) -> None:
        self._keywords = _KEYWORDS

    # ------------------------------------------------------------------
    def route(self, prompt: str) -> tuple[str, float]:
        """Return (best_domain, score). Falls back to ("general", 0.0)."""
        tokens = set(re.findall(r"[a-z]+", prompt.lower()))
        best_domain = "general"
        best_score = 0.0
        for domain, kws in self._keywords.items():
            matched = sum(1 for w in kws if w in tokens)
            score = matched / len(kws) if kws else 0.0
            if score > best_score:
                best_score = score
                best_domain = domain
        if best_score < 0.05:
            return "general", 0.0
        return best_domain, best_score

    # ------------------------------------------------------------------
    def route_task(self, prompt: str, explicit_domain: str | None = None) -> str:
        """Return explicit_domain if set and not 'general', else auto-route."""
        if explicit_domain and explicit_domain != "general":
            return explicit_domain
        domain, _ = self.route(prompt)
        return domain

    # ------------------------------------------------------------------
    def competition_scores(self, prompt: str) -> dict[str, float]:
        """Return score for every domain plus general, sorted descending."""
        tokens = set(re.findall(r"[a-z]+", prompt.lower()))
        scores: dict[str, float] = {}
        for domain, kws in self._keywords.items():
            matched = sum(1 for w in kws if w in tokens)
            scores[domain] = matched / len(kws) if kws else 0.0
        scores["general"] = 0.0
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
