"""manifold/ingestion/ingestion_router.py — Universal ingestion router.

UniversalIngester accepts any input (file path, URL, text, bytes) and routes
it to the correct specialist ingester: DocumentIngester, ImageIngester, or
AudioIngester.  Also maintains an in-memory history of the last 20 events.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path


_HISTORY: deque[dict] = deque(maxlen=20)
_HISTORY_LOCK = threading.Lock()


class UniversalIngester:
    """Route any input to the appropriate MANIFOLD ingester.

    Parameters
    ----------
    org_id:
        MANIFOLD org.
    model_endpoint:
        LLM endpoint shared across all ingesters.
    api_key:
        Bearer token.
    apply_rules:
        Whether to apply extracted rules to the live engine.
    """

    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
    _AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac"})
    _DOC_EXTS = frozenset({
        ".pdf", ".txt", ".text", ".md", ".markdown",
        ".csv", ".tsv", ".docx",
    })

    def __init__(
        self,
        org_id: str = "default",
        model_endpoint: str = "http://localhost:8080/v1/chat/completions",
        api_key: str = "",
        apply_rules: bool = True,
    ) -> None:
        self.org_id = org_id
        self.model_endpoint = model_endpoint
        self.api_key = api_key
        self.apply_rules = apply_rules

    def ingest(
        self,
        source: str | Path | bytes,
        mime_type: str = "",
    ) -> dict:
        """Ingest any source and return a unified result dict.

        Parameters
        ----------
        source:
            A URL (starts with http/https), a file path, raw text, or bytes.
        mime_type:
            Override MIME type detection when source is bytes.
        """
        result: dict = {
            "source_type": "unknown",
            "items_extracted": 0,
            "applied": 0,
            "errors": [],
            "timestamp": time.time(),
        }

        try:
            if isinstance(source, bytes):
                result = self._route_bytes(source, mime_type)
            elif isinstance(source, Path):
                result = self._route_path(source)
            else:
                source_str = str(source)
                if source_str.lower().startswith(("http://", "https://")):
                    result = self._route_url(source_str)
                elif Path(source_str).exists():
                    result = self._route_path(Path(source_str))
                else:
                    result = self._route_text(source_str)
        except Exception as exc:  # noqa: BLE001
            result["errors"] = [str(exc)]

        # Record in history
        with _HISTORY_LOCK:
            _HISTORY.append(dict(result))

        return result

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route_path(self, path: Path) -> dict:
        suffix = path.suffix.lower()
        if suffix in self._IMAGE_EXTS:
            return self._run_image(path)
        if suffix in self._AUDIO_EXTS:
            return self._run_audio(path)
        return self._run_document_file(path)

    def _route_url(self, url: str) -> dict:
        from manifold.ingestion.document_ingester import DocumentIngester  # noqa: PLC0415
        di = DocumentIngester(
            org_id=self.org_id,
            apply_rules=self.apply_rules,
        )
        r = di.ingest_url(url)
        return {
            "source_type": "url",
            "items_extracted": r.policies_extracted,
            "applied": r.policies_applied,
            "errors": r.errors,
            "format_detected": r.format_detected,
            "raw_text_preview": r.raw_text_preview,
            "timestamp": r.timestamp,
        }

    def _route_text(self, text: str) -> dict:
        from manifold.ingestion.document_ingester import DocumentIngester  # noqa: PLC0415
        di = DocumentIngester(
            org_id=self.org_id,
            apply_rules=self.apply_rules,
        )
        r = di.ingest_text(text, source="inline_text")
        return {
            "source_type": "text",
            "items_extracted": r.policies_extracted,
            "applied": r.policies_applied,
            "errors": r.errors,
            "format_detected": r.format_detected,
            "raw_text_preview": r.raw_text_preview,
            "timestamp": r.timestamp,
        }

    def _route_bytes(self, data: bytes, mime_type: str) -> dict:
        if "image" in mime_type:
            from manifold.ingestion.image_ingester import ImageIngester  # noqa: PLC0415
            ii = ImageIngester(
                org_id=self.org_id,
                model_endpoint=self.model_endpoint,
                api_key=self.api_key,
            )
            r = ii.ingest(data)
            return {
                "source_type": "image_bytes",
                "items_extracted": r.items_extracted,
                "applied": r.applied,
                "errors": r.errors,
                "pipeline_used": r.pipeline_used,
                "timestamp": time.time(),
            }
        if "audio" in mime_type:
            suffix = ".wav"
            if "mpeg" in mime_type or "mp3" in mime_type:
                suffix = ".mp3"
            from manifold.ingestion.audio_ingester import AudioIngester  # noqa: PLC0415
            ai = AudioIngester(
                org_id=self.org_id,
                model_endpoint=self.model_endpoint,
                api_key=self.api_key,
                apply_rules=self.apply_rules,
            )
            r = ai.ingest_bytes(data, suffix=suffix)
            return {
                "source_type": "audio_bytes",
                "items_extracted": r.policies_extracted,
                "applied": r.policies_applied,
                "errors": r.errors,
                "transcript": r.transcript[:200],
                "timestamp": r.timestamp,
            }
        # Treat as text
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = data.decode("latin-1", errors="replace")
        return self._route_text(text)

    def _run_document_file(self, path: Path) -> dict:
        from manifold.ingestion.document_ingester import DocumentIngester  # noqa: PLC0415
        di = DocumentIngester(
            org_id=self.org_id,
            apply_rules=self.apply_rules,
        )
        r = di.ingest_file(path)
        return {
            "source_type": "document",
            "items_extracted": r.policies_extracted,
            "applied": r.policies_applied,
            "errors": r.errors,
            "format_detected": r.format_detected,
            "raw_text_preview": r.raw_text_preview,
            "timestamp": r.timestamp,
        }

    def _run_image(self, path: Path) -> dict:
        from manifold.ingestion.image_ingester import ImageIngester  # noqa: PLC0415
        ii = ImageIngester(
            org_id=self.org_id,
            model_endpoint=self.model_endpoint,
            api_key=self.api_key,
        )
        r = ii.ingest(path)
        return {
            "source_type": "image",
            "items_extracted": r.items_extracted,
            "applied": r.applied,
            "errors": r.errors,
            "pipeline_used": r.pipeline_used,
            "timestamp": time.time(),
        }

    def _run_audio(self, path: Path) -> dict:
        from manifold.ingestion.audio_ingester import AudioIngester  # noqa: PLC0415
        ai = AudioIngester(
            org_id=self.org_id,
            model_endpoint=self.model_endpoint,
            api_key=self.api_key,
            apply_rules=self.apply_rules,
        )
        r = ai.ingest(path)
        return {
            "source_type": "audio",
            "items_extracted": r.policies_extracted,
            "applied": r.policies_applied,
            "errors": r.errors,
            "transcript": r.transcript[:200],
            "timestamp": r.timestamp,
        }


def get_ingest_history() -> list[dict]:
    """Return the last 20 ingestion events."""
    with _HISTORY_LOCK:
        return list(_HISTORY)
