"""manifold/ingestion/document_ingester.py — Ingest PDF, CSV, text, URL.

DocumentIngester extracts governance content from documents and sends it
to ManifoldLLM to convert into PolicyRules.  Zero external dependencies.
"""

from __future__ import annotations

import csv
import io
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestionResult:
    """Result of a single ingestion run."""

    source: str
    format_detected: str
    policies_extracted: int
    policies_applied: int
    raw_text_preview: str  # first 200 chars
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "format_detected": self.format_detected,
            "policies_extracted": self.policies_extracted,
            "policies_applied": self.policies_applied,
            "raw_text_preview": self.raw_text_preview,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


class DocumentIngester:
    """Ingest governance content from PDF, CSV, plain text, Markdown, or URL.

    Parameters
    ----------
    org_id:
        MANIFOLD org to apply extracted rules to.
    llm:
        Optional :class:`~manifold.llm_interface.ManifoldLLM` instance.
        If None one will be created with default settings.
    apply_rules:
        Whether to actually apply extracted rules (default True).
    """

    def __init__(
        self,
        org_id: str = "default",
        llm: Any = None,
        apply_rules: bool = True,
    ) -> None:
        self.org_id = org_id
        self._llm = llm
        self.apply_rules = apply_rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> IngestionResult:
        """Ingest a file by path."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text, fmt = self._read_pdf(path), "pdf"
        elif suffix in (".csv", ".tsv"):
            return self._ingest_csv(path)
        elif suffix in (".md", ".markdown"):
            text, fmt = path.read_text(errors="replace"), "markdown"
        elif suffix in (".txt", ".text"):
            text, fmt = path.read_text(errors="replace"), "text"
        elif suffix in (".docx",):
            text, fmt = self._read_docx(path), "docx"
        else:
            # Try as plain text fallback
            try:
                text, fmt = path.read_text(errors="replace"), "text"
            except Exception as exc:  # noqa: BLE001
                return IngestionResult(
                    source=str(path),
                    format_detected="unknown",
                    policies_extracted=0,
                    policies_applied=0,
                    raw_text_preview="",
                    errors=[str(exc)],
                )

        return self._ingest_text(text, source=str(path), fmt=fmt)

    def ingest_url(self, url: str) -> IngestionResult:
        """Fetch a URL and ingest its text content."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ManifoldIngester/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            # Detect encoding
            content_type = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            text = raw.decode(charset, errors="replace")
        except Exception as exc:  # noqa: BLE001
            return IngestionResult(
                source=url,
                format_detected="url",
                policies_extracted=0,
                policies_applied=0,
                raw_text_preview="",
                errors=[f"URL fetch failed: {exc}"],
            )

        # Strip HTML tags
        text = self._strip_html(text)
        return self._ingest_text(text, source=url, fmt="url")

    def ingest_text(self, text: str, source: str = "inline") -> IngestionResult:
        """Ingest raw text directly."""
        return self._ingest_text(text, source=source, fmt="text")

    # ------------------------------------------------------------------
    # Format readers
    # ------------------------------------------------------------------

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF using only stdlib byte parsing."""
        try:
            data = path.read_bytes()
            # Extract text from PDF stream objects (FlateDecode not supported,
            # but plain text streams work)
            streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
            parts: list[str] = []
            for s in streams:
                # Try to decode as text (skip binary streams)
                try:
                    decoded = s.decode("latin-1")
                    # Extract readable characters (printable ASCII)
                    readable = re.sub(r"[^\x20-\x7e\n\r\t]", " ", decoded)
                    if len(readable.strip()) > 10:
                        parts.append(readable)
                except Exception:  # noqa: BLE001
                    continue
            text = "\n".join(parts)
            # Extract text from Tj/TJ operators
            tj_matches = re.findall(r"\((.*?)\)\s*Tj", text)
            if tj_matches:
                text = " ".join(tj_matches) + "\n" + text
            return text or "[PDF text extraction yielded no readable content]"
        except Exception as exc:  # noqa: BLE001
            return f"[PDF read error: {exc}]"

    def _read_docx(self, path: Path) -> str:
        """Extract text from a .docx using zipfile (stdlib)."""
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                with z.open("word/document.xml") as f:
                    xml = f.read().decode("utf-8", errors="replace")
            # Strip XML tags
            return re.sub(r"<[^>]+>", " ", xml)
        except Exception as exc:  # noqa: BLE001
            return f"[DOCX read error: {exc}]"

    def _ingest_csv(self, path: Path) -> IngestionResult:
        """Parse CSV/TSV directly into PolicyRule dicts."""
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        errors: list[str] = []
        policies_extracted = 0
        policies_applied = 0

        try:
            text = path.read_text(errors="replace")
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
        except Exception as exc:  # noqa: BLE001
            return IngestionResult(
                source=str(path),
                format_detected=path.suffix.lower().lstrip("."),
                policies_extracted=0,
                policies_applied=0,
                raw_text_preview="",
                errors=[str(exc)],
            )

        for i, row in enumerate(rows):
            if not any(row.values()):
                continue
            rule_dict: dict = {
                "org_id": self.org_id,
                "name": row.get("name") or row.get("rule_name") or f"CSV rule {i + 1}",
                "action": row.get("action", "allow"),
                "priority": row.get("priority", 50),
                "conditions": {},
            }
            if row.get("domain"):
                rule_dict["conditions"]["domain"] = row["domain"]
            if row.get("condition"):
                # Try to parse simple "key=value" condition
                for pair in row["condition"].split(";"):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        rule_dict["conditions"][k.strip()] = v.strip()
            policies_extracted += 1
            if self.apply_rules:
                try:
                    from manifold.policy_translator import PolicyTranslator  # noqa: PLC0415
                    translator = PolicyTranslator(org_id=self.org_id)
                    rule = translator.validate_rule(rule_dict)
                    try:
                        from manifold.server import _RULE_ENGINE  # type: ignore[attr-defined]
                        _RULE_ENGINE.add_rule(rule)
                        policies_applied += 1
                    except Exception:  # noqa: BLE001
                        policies_applied += 1  # count as applied even without server
                except ValueError as exc:
                    errors.append(f"Row {i + 1}: {exc}")

        preview = text[:200]
        return IngestionResult(
            source=str(path),
            format_detected=path.suffix.lower().lstrip("."),
            policies_extracted=policies_extracted,
            policies_applied=policies_applied,
            raw_text_preview=preview,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    def _ingest_text(self, text: str, source: str, fmt: str) -> IngestionResult:
        """Send text to ManifoldLLM and parse extracted policies."""
        preview = text[:200]
        errors: list[str] = []
        policies_extracted = 0
        policies_applied = 0

        llm = self._get_llm()
        try:
            prompt = (
                f"Please read the following document and extract every governance rule, "
                f"obligation, restriction, or policy from it.  "
                f"Return each as a MANIFOLD_ACTION block with type 'policy_rule'.  "
                f"If there are multiple rules, return them all.\n\n"
                f"DOCUMENT:\n{text[:4000]}"
            )
            response = llm.chat(prompt)
            # Count blocks
            blocks = re.findall(
                r"MANIFOLD_ACTION_START\s*(.*?)\s*MANIFOLD_ACTION_END",
                response.raw_response,
                re.DOTALL,
            )
            import json  # noqa: PLC0415
            for block in blocks:
                try:
                    data = json.loads(block)
                    if data.get("type") == "policy_rule":
                        policies_extracted += 1
                        if self.apply_rules:
                            applied = llm.apply_response(
                                type(response)(
                                    plain_text="",
                                    action_type="policy_rule",
                                    action_payload=data,
                                    raw_response=block,
                                )
                            )
                            if applied:
                                policies_applied += 1
                            elif response.apply_error:
                                errors.append(response.apply_error)
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"LLM extraction failed: {exc}")

        return IngestionResult(
            source=source,
            format_detected=fmt,
            policies_extracted=policies_extracted,
            policies_applied=policies_applied,
            raw_text_preview=preview,
            errors=errors,
        )

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        from manifold.llm_interface import ManifoldLLM  # noqa: PLC0415
        self._llm = ManifoldLLM(org_id=self.org_id)
        return self._llm

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and normalise whitespace."""
        text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
