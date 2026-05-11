"""manifold/ingestion — Universal policy ingestion from any source."""

from manifold.ingestion.document_ingester import DocumentIngester, IngestionResult
from manifold.ingestion.image_ingester import ImageIngester
from manifold.ingestion.audio_ingester import AudioIngester, TranscriptionResult
from manifold.ingestion.ingestion_router import UniversalIngester

__all__ = [
    "DocumentIngester",
    "IngestionResult",
    "ImageIngester",
    "AudioIngester",
    "TranscriptionResult",
    "UniversalIngester",
]
