"""tests/test_audio_ingester.py — Tests for AudioIngester."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manifold.ingestion.audio_ingester import AudioIngester, TranscriptionResult

FIXTURES = Path(__file__).parent / "fixtures"


def _make_ingester(transcript: str = "", llm_response: str = "") -> AudioIngester:
    """Return an AudioIngester with transcription and LLM mocked."""
    ai = AudioIngester(org_id="test-org", apply_rules=False)
    # Mock whisper-path to return our transcript
    ai._transcribe_whisper = MagicMock(return_value=(transcript, None))
    ai._transcribe_llm_fallback = MagicMock(return_value=(transcript, None, ""))

    # Mock _extract_policies
    if llm_response:
        ai._extract_policies = MagicMock(return_value=(1, 1, []))
    else:
        ai._extract_policies = MagicMock(return_value=(0, 0, []))
    return ai


def test_ingest_wav_returns_result():
    """Ingesting a WAV file returns a TranscriptionResult with a transcript."""
    ai = _make_ingester(transcript="Please audit all finance operations.")
    # Patch _WHISPER_AVAILABLE to True for this test
    with patch("manifold.ingestion.audio_ingester._WHISPER_AVAILABLE", True):
        result = ai.ingest(FIXTURES / "test_audio.wav")
    assert isinstance(result, TranscriptionResult)
    assert result.transcript == "Please audit all finance operations."


def test_no_whisper_uses_fallback():
    """When whisper is not available, the LLM fallback is attempted."""
    ai = AudioIngester(org_id="test-org", apply_rules=False)
    ai._transcribe_llm_fallback = MagicMock(
        return_value=("Fallback transcript.", None, "")
    )
    ai._extract_policies = MagicMock(return_value=(0, 0, []))
    with patch("manifold.ingestion.audio_ingester._WHISPER_AVAILABLE", False):
        result = ai.ingest(FIXTURES / "test_audio.wav")
    assert isinstance(result, TranscriptionResult)
    ai._transcribe_llm_fallback.assert_called_once()


def test_ingest_bytes_works():
    """ingest_bytes should write to a temp file and return a result."""
    raw = (FIXTURES / "test_audio.wav").read_bytes()
    ai = _make_ingester(transcript="Voice policy test.")
    with patch("manifold.ingestion.audio_ingester._WHISPER_AVAILABLE", True):
        result = ai.ingest_bytes(raw, suffix=".wav")
    assert isinstance(result, TranscriptionResult)
