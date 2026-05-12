"""manifold/routes/ingestion.py — Document/image/audio ingestion handlers.

Handlers for:
  POST /ingest/document
  POST /ingest/image
  POST /ingest/audio
  POST /ingest
  GET  /ingest/history
  POST /rules/preset
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


def handle_post_ingest_document(self: "ManifoldHandler", body: dict) -> None:
    """POST /ingest/document — ingest a URL or text body as governance document."""
    from manifold.ingestion.document_ingester import DocumentIngester  # noqa: PLC0415
    s = _srv()
    org_id = body.get("org_id", "default")
    di = DocumentIngester(org_id=org_id, apply_rules=True)
    url = body.get("url") or body.get("source") or ""
    text = body.get("text") or ""
    if url:
        result = di.ingest_url(url)
    elif text:
        result = di.ingest_text(text)
    else:
        s._send_error(self, 400, "Body must include 'url' or 'text' field.")
        return
    s._send_json(self, 200, result.to_dict())


def handle_post_ingest_image(self: "ManifoldHandler", body: dict) -> None:
    """POST /ingest/image — ingest image bytes (base64-encoded)."""
    import base64  # noqa: PLC0415
    from manifold.ingestion.image_ingester import ImageIngester  # noqa: PLC0415
    s = _srv()
    org_id = body.get("org_id", "default")
    pipeline = body.get("pipeline", "auto")
    ii = ImageIngester(
        org_id=org_id,
        model_endpoint=body.get("model_endpoint", "http://localhost:8080/v1/chat/completions"),
        api_key=body.get("api_key", ""),
    )
    b64data = body.get("image_b64") or body.get("data") or ""
    if not b64data:
        s._send_error(self, 400, "Body must include 'image_b64' field (base64-encoded image).")
        return
    try:
        image_bytes = base64.b64decode(b64data)
    except Exception as exc:  # noqa: BLE001
        s._send_error(self, 400, f"Invalid base64 data: {exc}")
        return
    result = ii.ingest(image_bytes, pipeline=pipeline)
    s._send_json(self, 200, result.to_dict())


def handle_post_ingest_audio(self: "ManifoldHandler", body: dict) -> None:
    """POST /ingest/audio — ingest base64-encoded audio."""
    import base64  # noqa: PLC0415
    from manifold.ingestion.audio_ingester import AudioIngester  # noqa: PLC0415
    s = _srv()
    org_id = body.get("org_id", "default")
    ai = AudioIngester(
        org_id=org_id,
        model_endpoint=body.get("model_endpoint", "http://localhost:8080/v1/chat/completions"),
        api_key=body.get("api_key", ""),
        apply_rules=True,
    )
    b64data = body.get("audio_b64") or body.get("data") or ""
    suffix = body.get("format", ".wav")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if not b64data:
        s._send_error(self, 400, "Body must include 'audio_b64' field (base64-encoded audio).")
        return
    try:
        audio_bytes = base64.b64decode(b64data)
    except Exception as exc:  # noqa: BLE001
        s._send_error(self, 400, f"Invalid base64 data: {exc}")
        return
    result = ai.ingest_bytes(audio_bytes, suffix=suffix)
    s._send_json(self, 200, result.to_dict())


def handle_post_ingest(self: "ManifoldHandler", body: dict) -> None:
    """POST /ingest — universal ingestion, auto-detects content type."""
    from manifold.ingestion.ingestion_router import UniversalIngester  # noqa: PLC0415
    s = _srv()
    org_id = body.get("org_id", "default")
    ui = UniversalIngester(
        org_id=org_id,
        model_endpoint=body.get("model_endpoint", "http://localhost:8080/v1/chat/completions"),
        api_key=body.get("api_key", ""),
        apply_rules=True,
    )
    url = body.get("url") or ""
    text = body.get("text") or ""
    if url:
        result = ui.ingest(url)
    elif text:
        result = ui.ingest(text)
    else:
        s._send_error(self, 400, "Body must include 'url' or 'text' field.")
        return
    s._send_json(self, 200, result)


def handle_get_ingest_history(self: "ManifoldHandler") -> None:
    """GET /ingest/history — last 20 ingestion events."""
    from manifold.ingestion.ingestion_router import get_ingest_history  # noqa: PLC0415
    s = _srv()
    s._send_json(self, 200, {"history": get_ingest_history()})


def handle_post_rules_preset(self: "ManifoldHandler", body: dict) -> None:
    """POST /rules/preset — apply a compliance preset (hipaa/gdpr/sox/iso27001)."""
    from manifold.policy_translator import PolicyTranslator  # noqa: PLC0415
    s = _srv()
    preset = body.get("preset") or body.get("name") or ""
    if not preset:
        s._send_error(self, 400, "Body must include a 'preset' field.")
        return
    org_id = body.get("org_id", "default")
    try:
        rules = PolicyTranslator.apply_preset(preset, org_id=org_id)
    except ValueError as exc:
        s._send_error(self, 400, str(exc))
        return
    applied = 0
    for rule in rules:
        try:
            s._RULE_ENGINE.add_rule(rule)
            applied += 1
        except Exception:  # noqa: BLE001
            pass
    s._send_json(self, 200, {
        "preset": preset,
        "rules_applied": applied,
        "rule_names": [r.name for r in rules],
    })
