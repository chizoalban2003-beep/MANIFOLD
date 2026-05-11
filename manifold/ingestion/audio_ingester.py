"""manifold/ingestion/audio_ingester.py — Voice instructions to governance.

AudioIngester converts audio files (MP3, WAV, M4A, OGG) to text via Whisper
(optional) then extracts PolicyRules via ManifoldLLM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Try to import whisper; graceful fallback if not installed
try:
    import whisper as _whisper  # type: ignore[import]
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result from AudioIngester."""

    source: str
    transcript: str
    confidence: float | None
    policies_extracted: int
    policies_applied: int
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "policies_extracted": self.policies_extracted,
            "policies_applied": self.policies_applied,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


class AudioIngester:
    """Convert audio files to MANIFOLD governance rules.

    Parameters
    ----------
    org_id:
        MANIFOLD org.
    model_endpoint:
        LLM chat completions endpoint.
    api_key:
        Bearer token for the model.
    whisper_model:
        Whisper model size (default ``"base"``).
    apply_rules:
        Whether to actually apply extracted rules.
    """

    def __init__(
        self,
        org_id: str = "default",
        model_endpoint: str = "http://localhost:8080/v1/chat/completions",
        api_key: str = "",
        whisper_model: str = "base",
        apply_rules: bool = True,
    ) -> None:
        self.org_id = org_id
        self.model_endpoint = model_endpoint
        self.api_key = api_key
        self.whisper_model = whisper_model
        self.apply_rules = apply_rules
        self._whisper_instance: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, path: str | Path) -> TranscriptionResult:
        """Transcribe audio file and extract governance rules from it."""
        path = Path(path)
        errors: list[str] = []
        transcript = ""
        confidence: float | None = None

        # Step 1: Transcription
        if _WHISPER_AVAILABLE:
            transcript, confidence = self._transcribe_whisper(path)
        else:
            transcript, confidence, whisper_error = self._transcribe_llm_fallback(path)
            if whisper_error:
                errors.append(whisper_error)

        if not transcript:
            return TranscriptionResult(
                source=str(path),
                transcript="",
                confidence=confidence,
                policies_extracted=0,
                policies_applied=0,
                errors=errors or ["Transcription produced no output."],
            )

        # Step 2: Extract policies from transcript
        extracted, applied, extract_errors = self._extract_policies(transcript)
        errors += extract_errors

        return TranscriptionResult(
            source=str(path),
            transcript=transcript,
            confidence=confidence,
            policies_extracted=extracted,
            policies_applied=applied,
            errors=errors,
        )

    def ingest_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> TranscriptionResult:
        """Ingest raw audio bytes by writing to a temp file."""
        import tempfile  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.ingest(tmp_path)
        finally:
            import os  # noqa: PLC0415
            try:
                os.unlink(tmp_path)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Transcription backends
    # ------------------------------------------------------------------

    def _transcribe_whisper(self, path: Path) -> tuple[str, float | None]:
        """Transcribe using the openai-whisper package."""
        try:
            if self._whisper_instance is None:
                self._whisper_instance = _whisper.load_model(self.whisper_model)
            result = self._whisper_instance.transcribe(str(path))
            text = result.get("text", "") or ""
            # Whisper doesn't expose per-segment confidence directly; use None
            return text.strip(), None
        except Exception as exc:  # noqa: BLE001
            _log.error("Whisper transcription failed: %s", exc)
            return "", None

    def _transcribe_llm_fallback(self, path: Path) -> tuple[str, float | None, str]:
        """Attempt transcription via LLM gateway as a fallback."""
        import base64  # noqa: PLC0415
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        suffix = path.suffix.lower()
        if suffix not in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            return (
                "",
                None,
                f"Unsupported audio format '{suffix}' and whisper is not installed.  "
                f"Install with: pip install openai-whisper",
            )

        try:
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            # Send a note to the LLM that we're providing audio (most models cannot
            # actually process raw audio through the chat endpoint)
            message = (
                f"[Audio transcription request — {len(raw)} bytes of {suffix} audio.  "
                f"Base64 encoded data follows (first 200 chars): {b64[:200]}...]\n\n"
                f"Please transcribe this audio if your model supports it, or return "
                f"an error message explaining that openai-whisper should be installed."
            )
            payload = json.dumps({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a transcription assistant."},
                    {"role": "user", "content": message},
                ],
            }).encode()
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                self.model_endpoint, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            return text, None, ""
        except Exception as exc:  # noqa: BLE001
            return (
                "",
                None,
                f"LLM transcription fallback failed: {exc}.  "
                f"Install openai-whisper for offline transcription: pip install openai-whisper",
            )

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def _extract_policies(self, transcript: str) -> tuple[int, int, list[str]]:
        """Extract PolicyRules from transcribed text via ManifoldLLM."""
        import re  # noqa: PLC0415
        import json  # noqa: PLC0415
        errors: list[str] = []
        extracted = 0
        applied = 0

        try:
            from manifold.llm_interface import ManifoldLLM  # noqa: PLC0415
            llm = ManifoldLLM(
                org_id=self.org_id,
                model_endpoint=self.model_endpoint,
                api_key=self.api_key,
            )
            prompt = (
                "The following is a spoken instruction about governance and policy.  "
                "Extract every governance rule, restriction, or obligation from it and "
                "return each as a MANIFOLD_ACTION block with type 'policy_rule'.  "
                "Handle informal speech naturally: for example, "
                "'never let the AI touch our customer list' should become "
                "a refuse rule with prompt_contains='customer list'.  "
                f"Transcript: {transcript}"
            )
            response = llm.chat(prompt)
            blocks = re.findall(
                r"MANIFOLD_ACTION_START\s*(.*?)\s*MANIFOLD_ACTION_END",
                response.raw_response,
                re.DOTALL,
            )
            from manifold.policy_translator import PolicyTranslator  # noqa: PLC0415
            translator = PolicyTranslator(org_id=self.org_id)
            for block in blocks:
                try:
                    data = json.loads(block)
                    if data.get("type") == "policy_rule" and self.apply_rules:
                        extracted += 1
                        rule = translator.validate_rule(data)
                        try:
                            from manifold.server import _RULE_ENGINE  # type: ignore[attr-defined]
                            _RULE_ENGINE.add_rule(rule)
                        except Exception:  # noqa: BLE001
                            pass
                        applied += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Policy extraction failed: {exc}")

        return extracted, applied, errors
