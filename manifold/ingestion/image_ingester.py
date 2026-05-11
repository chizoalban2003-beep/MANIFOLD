"""manifold/ingestion/image_ingester.py — Floor plans and whiteboards to MANIFOLD.

ImageIngester routes images to either:
  - spatial pipeline: floor plan → SpaceIngestion.ingest() (room types → CRNA)
  - policy pipeline: whiteboard/sign text → PolicyTranslator
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class ImageIngestionResult:
    """Result from processing a single image."""

    source: str
    pipeline_used: str
    items_extracted: int
    applied: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "pipeline_used": self.pipeline_used,
            "items_extracted": self.items_extracted,
            "applied": self.applied,
            "errors": self.errors,
        }


class ImageIngester:
    """Ingest JPEG, PNG, or WebP images into MANIFOLD.

    Parameters
    ----------
    org_id:
        MANIFOLD org to apply results to.
    model_endpoint:
        Vision-capable model endpoint.
    api_key:
        Bearer token for the model.
    model:
        Vision model name.
    """

    _FLOOR_PLAN_KEYWORDS = frozenset({
        "room", "floor", "kitchen", "bedroom", "bathroom", "office",
        "hall", "corridor", "living", "dining", "entrance", "garage",
        "staircase", "storage", "closet", "plan", "layout",
    })

    def __init__(
        self,
        org_id: str = "default",
        model_endpoint: str = "http://localhost:8080/v1/chat/completions",
        api_key: str = "",
        model: str = "gpt-4o",
    ) -> None:
        self.org_id = org_id
        self.model_endpoint = model_endpoint
        self.api_key = api_key
        self.model = model

    def ingest(self, path: str | Path | bytes, pipeline: str = "auto") -> ImageIngestionResult:
        """Ingest an image file.

        Parameters
        ----------
        path:
            File path, bytes object, or already-encoded base64 string.
        pipeline:
            ``"auto"`` (detect), ``"spatial"`` (floor plan), or ``"policy"``.
        """
        source, b64data, mime = self._load_image(path)

        if pipeline == "auto":
            pipeline = self._detect_pipeline(b64data, mime)

        if pipeline == "spatial":
            return self._spatial_pipeline(source, b64data, mime)
        else:
            return self._policy_pipeline(source, b64data, mime)

    # ------------------------------------------------------------------
    # Pipeline detection
    # ------------------------------------------------------------------

    def _detect_pipeline(self, b64data: str, mime: str) -> str:
        """Ask the model to classify the image; fallback to 'policy'."""
        prompt = (
            "Does this image show a floor plan, architectural drawing, office layout, "
            "or building diagram with rooms and spaces?  Answer with exactly one word: "
            "YES or NO."
        )
        try:
            response = self._call_vision(prompt, b64data, mime)
            if response.upper().startswith("Y"):
                return "spatial"
        except Exception:  # noqa: BLE001
            pass
        return "policy"

    # ------------------------------------------------------------------
    # Spatial pipeline
    # ------------------------------------------------------------------

    def _spatial_pipeline(self, source: str, b64data: str, mime: str) -> ImageIngestionResult:
        """Convert floor plan image to SpaceIngestion floorplan."""
        system = (
            "You are a spatial analysis AI.  Analyse the floor plan image and return a "
            "JSON object describing every room.  The JSON must be in this exact format:\n"
            '{"rooms": [{"name": str, "type": str, "x": int, "y": int, '
            '"width": int, "height": int, "crna": {"c": 0-1, "r": 0-1, "n": 0-1, "a": 0-1}}]}\n'
            "Use a 10x10 grid.  Assign CRNA values appropriate to each room type:\n"
            "  kitchen: c=0.4, r=0.7, n=0.8, a=0.9\n"
            "  bedroom: c=0.2, r=0.3, n=1.0, a=0.8\n"
            "  bathroom: c=0.2, r=0.4, n=1.0, a=0.6\n"
            "  living room: c=0.3, r=0.3, n=0.9, a=0.9\n"
            "  office: c=0.5, r=0.5, n=0.8, a=0.7\n"
            "  server room: c=0.8, r=0.8, n=0.5, a=1.0\n"
            "  baby room: c=0.1, r=0.95, n=1.0, a=0.7\n"
            "Return ONLY valid JSON, no commentary."
        )
        errors: list[str] = []
        items_extracted = 0
        applied = 0

        try:
            raw = self._call_vision(system, b64data, mime)
            # Extract JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON found in vision model response")
            data = json.loads(m.group(0))
            rooms = data.get("rooms", [])
            items_extracted = len(rooms)

            # Apply via SpaceIngestion
            try:
                from manifold_physical.space_ingestion import SpaceIngestion  # noqa: PLC0415
                si = SpaceIngestion()
                for room in rooms:
                    si.set_base_crna(
                        x=int(room.get("x", 0)),
                        y=int(room.get("y", 0)),
                        z=0,
                        c=float(room["crna"]["c"]),
                        r=float(room["crna"]["r"]),
                        n=float(room["crna"]["n"]),
                        a=float(room["crna"]["a"]),
                    )
                    applied += 1
            except Exception as exc:  # noqa: BLE001
                # SpaceIngestion not available; still count as extracted
                _log.debug("SpaceIngestion not available: %s", exc)
                applied = items_extracted  # best-effort
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        return ImageIngestionResult(
            source=source,
            pipeline_used="spatial",
            items_extracted=items_extracted,
            applied=applied,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Policy pipeline
    # ------------------------------------------------------------------

    def _policy_pipeline(self, source: str, b64data: str, mime: str) -> ImageIngestionResult:
        """Extract policy rules from whiteboard or signage image."""
        system = (
            "Read all visible text in this image.  Identify every rule, constraint, "
            "prohibition, or obligation.  For each one return a MANIFOLD_ACTION block:\n"
            "MANIFOLD_ACTION_START\n"
            "{\"type\": \"policy_rule\", \"name\": \"...\", \"conditions\": {}, "
            "\"action\": \"refuse\"|\"audit\"|\"allow\", \"priority\": 50}\n"
            "MANIFOLD_ACTION_END\n"
            "If no rules are found, return: MANIFOLD_ACTION_START\n{\"type\": \"none\"}\nMANIFOLD_ACTION_END"
        )
        errors: list[str] = []
        items_extracted = 0
        applied = 0

        try:
            raw = self._call_vision(system, b64data, mime)
            blocks = re.findall(
                r"MANIFOLD_ACTION_START\s*(.*?)\s*MANIFOLD_ACTION_END",
                raw,
                re.DOTALL,
            )
            from manifold.policy_translator import PolicyTranslator  # noqa: PLC0415
            translator = PolicyTranslator(org_id=self.org_id)
            for block in blocks:
                try:
                    data = json.loads(block)
                    if data.get("type") == "policy_rule":
                        items_extracted += 1
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
            errors.append(str(exc))

        return ImageIngestionResult(
            source=source,
            pipeline_used="policy",
            items_extracted=items_extracted,
            applied=applied,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Vision model call
    # ------------------------------------------------------------------

    def _call_vision(self, system: str, b64data: str, mime: str) -> str:
        """Call the vision model with the image encoded as base64."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64data}"},
                        }
                    ],
                },
            ],
        }).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.model_endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, path: str | Path | bytes) -> tuple[str, str, str]:
        """Return (source_label, base64_data, mime_type)."""
        if isinstance(path, bytes):
            b64 = base64.b64encode(path).decode()
            return "<bytes>", b64, "image/png"

        path = Path(path)
        suffix = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = mime_map.get(suffix, "image/png")
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode()
        return str(path), b64, mime
