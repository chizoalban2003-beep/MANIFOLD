"""manifold/encoders — unified encoding dispatch layer."""
from __future__ import annotations

from manifold.encoder_v2 import EncodedTask, encode_prompt
from .timeseries import TimeSeriesEncoder
from .structured import StructuredEncoder

__all__ = ["encode_any", "EncodedTask", "TimeSeriesEncoder", "StructuredEncoder"]


def encode_any(data: object, encoder_hint: str = "auto") -> EncodedTask:
    """Dispatch to the right encoder based on hint or data type."""
    if encoder_hint == "timeseries" or (encoder_hint == "auto" and isinstance(data, list)):
        return TimeSeriesEncoder().encode(data)  # type: ignore[arg-type]
    if encoder_hint == "structured" or (encoder_hint == "auto" and isinstance(data, dict)):
        return StructuredEncoder().encode(data)  # type: ignore[arg-type]
    return encode_prompt(str(data), force_keyword=True)
