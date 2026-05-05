"""Phase 47: Native Vector File System — Pure-Python Semantic Memory.

AI agents need semantic memory.  This module provides a zero-dependency,
pure-Python vector database for storing and retrieving high-dimensional
embeddings.

Architecture
------------
1. :class:`VectorEntry` — stores a single embedding and its metadata.
2. :func:`cosine_similarity` — highly-optimised pure-Python cosine similarity.
3. :class:`LSHIndex` — random-projection Locality-Sensitive Hashing that
   partitions vectors into buckets, enabling sub-linear search.
4. :class:`VectorIndex` — the public API: ``add``, ``search``, ``bucket_summary``,
   and ``to_dicts`` for vault persistence.

Cosine similarity formula
-------------------------
.. math::

    S_C(A,B) = \\frac{\\sum_i A_i B_i}
                     {\\sqrt{\\sum_i A_i^2}\\,\\sqrt{\\sum_i B_i^2}}

A zero vector has similarity ``0.0`` with any other vector.

Key classes
-----------
``VectorEntry``
    A single stored embedding with optional metadata.
``LSHIndex``
    Random-projection LSH for O(1) bucket lookup.
``VectorIndex``
    Full vector database: add, LSH-accelerated search, serialisation.
``SearchResult``
    Immutable (vector_id, similarity) pair returned by :meth:`VectorIndex.search`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# VectorEntry
# ---------------------------------------------------------------------------


@dataclass
class VectorEntry:
    """A single stored vector with associated metadata.

    Attributes
    ----------
    vector_id:
        Unique identifier for this entry.
    vector:
        The embedding as a list of floats.
    metadata:
        Arbitrary key/value data attached to the entry.
    """

    vector_id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "vector_id": self.vector_id,
            "vector": self.vector,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Immutable nearest-neighbor result.

    Attributes
    ----------
    vector_id:
        Identifier of the matching entry.
    similarity:
        Cosine similarity score in ``[-1.0, 1.0]``.
    """

    vector_id: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"vector_id": self.vector_id, "similarity": self.similarity}


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Parameters
    ----------
    a, b:
        Vectors of the same dimension.

    Returns
    -------
    float
        Similarity in ``[-1.0, 1.0]``.  Returns ``0.0`` if either vector is
        a zero-vector (magnitude is 0).

    Raises
    ------
    ValueError
        If the two vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}"
        )

    dot = 0.0
    sq_a = 0.0
    sq_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        sq_a += ai * ai
        sq_b += bi * bi

    mag_a = math.sqrt(sq_a)
    mag_b = math.sqrt(sq_b)

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# LSHIndex
# ---------------------------------------------------------------------------


@dataclass
class LSHIndex:
    """Random-projection Locality-Sensitive Hashing.

    Each vector is hashed into a binary bucket key by projecting it onto
    ``n_planes`` random hyperplanes and recording the sign of each projection.
    Two vectors that are close in cosine distance will tend to land in the
    same bucket.

    Parameters
    ----------
    n_planes:
        Number of random hyperplanes (bits in the bucket key).
        Higher values → finer partitioning; lower values → more recall.
        Default: ``8``.
    seed:
        Optional integer seed for reproducible hyperplanes.

    Notes
    -----
    :meth:`fit` must be called (or triggered automatically via
    :class:`VectorIndex`) before any hashing.
    """

    n_planes: int = 8
    seed: int | None = None

    _dim: int = field(default=0, init=False, repr=False)
    _planes: list[list[float]] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def fit(self, dim: int) -> None:
        """Initialise random hyperplanes for *dim*-dimensional vectors.

        Parameters
        ----------
        dim:
            Dimensionality of the vectors to be indexed.
        """
        self._dim = dim
        rng = random.Random(self.seed)
        self._planes = [
            [rng.gauss(0.0, 1.0) for _ in range(dim)]
            for _ in range(self.n_planes)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensionality this index was fitted for (0 if not yet fitted)."""
        return self._dim

    def hash_vector(self, vector: list[float]) -> tuple[int, ...]:
        """Compute the LSH bucket key for *vector*.

        Parameters
        ----------
        vector:
            A vector of length :attr:`dim`.

        Returns
        -------
        tuple[int, ...]
            Binary tuple (``0``/``1`` per plane) used as a bucket key.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        ValueError
            If ``len(vector) != dim``.
        """
        if not self._planes:
            raise RuntimeError("LSHIndex has not been fitted; call fit(dim) first.")
        if len(vector) != self._dim:
            raise ValueError(
                f"Expected {self._dim}-dim vector, got {len(vector)}"
            )
        return tuple(
            1 if sum(v * p for v, p in zip(vector, plane)) >= 0.0 else 0
            for plane in self._planes
        )


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------


@dataclass
class VectorIndex:
    """Pure-Python vector database with cosine similarity and LSH search.

    Parameters
    ----------
    n_planes:
        Number of LSH hyperplanes.  Increasing this refines bucket
        granularity at the cost of recall (fewer vectors per bucket).
        Default: ``8``.
    seed:
        Optional RNG seed for reproducible LSH planes.

    Example
    -------
    ::

        idx = VectorIndex()
        idx.add("doc1", [0.1, 0.9, 0.0], metadata={"text": "hello"})
        idx.add("doc2", [0.8, 0.2, 0.0])
        results = idx.search([0.1, 0.85, 0.0], k=1)
        assert results[0].vector_id == "doc1"
    """

    n_planes: int = 8
    seed: int | None = None

    _dim: int = field(default=0, init=False, repr=False)
    _entries: dict[str, VectorEntry] = field(default_factory=dict, init=False, repr=False)
    _buckets: dict[tuple[int, ...], list[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lsh: LSHIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lsh = LSHIndex(n_planes=self.n_planes, seed=self.seed)

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def add(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or replace a vector in the index.

        Parameters
        ----------
        vector_id:
            Unique identifier for this vector.
        vector:
            The embedding to store.
        metadata:
            Optional arbitrary metadata attached to the entry.

        Raises
        ------
        ValueError
            If *vector* has a different dimension than previously added vectors.
        """
        if len(vector) == 0:
            raise ValueError("Cannot add a zero-length vector.")

        if self._dim == 0:
            self._dim = len(vector)
            self._lsh.fit(self._dim)
        elif len(vector) != self._dim:
            raise ValueError(
                f"Expected {self._dim}-dim vector, got {len(vector)}"
            )

        # Remove from old bucket if updating an existing entry
        if vector_id in self._entries:
            old_bucket = self._lsh.hash_vector(self._entries[vector_id].vector)
            bucket_list = self._buckets.get(old_bucket, [])
            if vector_id in bucket_list:
                bucket_list.remove(vector_id)

        entry = VectorEntry(
            vector_id=vector_id,
            vector=list(vector),
            metadata=dict(metadata) if metadata else {},
        )
        self._entries[vector_id] = entry
        bucket = self._lsh.hash_vector(vector)
        self._buckets.setdefault(bucket, []).append(vector_id)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        """Return the *k* most similar vectors to *query_vector*.

        The search uses the LSH bucket matching for O(1) pre-filtering:
        only vectors in the same bucket as the query are scored.  If the
        matching bucket is empty the search falls back to a full linear
        scan to guarantee at least one result (when the index is non-empty).

        Parameters
        ----------
        query_vector:
            The query embedding.
        k:
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Up to *k* results sorted by descending similarity.

        Raises
        ------
        ValueError
            If *query_vector* has a different dimension than the index.
        """
        if not self._entries:
            return []

        if self._dim == 0:
            return []

        if len(query_vector) != self._dim:
            raise ValueError(
                f"Expected {self._dim}-dim query, got {len(query_vector)}"
            )

        bucket = self._lsh.hash_vector(query_vector)
        candidate_ids = self._buckets.get(bucket, [])

        if not candidate_ids:
            # Fallback: full scan
            candidate_ids = list(self._entries.keys())

        scored: list[SearchResult] = []
        for vid in candidate_ids:
            entry = self._entries[vid]
            sim = cosine_similarity(query_vector, entry.vector)
            scored.append(SearchResult(vector_id=vid, similarity=sim))

        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:k]

    def get(self, vector_id: str) -> VectorEntry | None:
        """Return the entry for *vector_id*, or ``None`` if not found."""
        return self._entries.get(vector_id)

    def remove(self, vector_id: str) -> bool:
        """Remove *vector_id* from the index.

        Returns
        -------
        bool
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """
        entry = self._entries.pop(vector_id, None)
        if entry is None:
            return False
        bucket = self._lsh.hash_vector(entry.vector)
        bucket_list = self._buckets.get(bucket, [])
        if vector_id in bucket_list:
            bucket_list.remove(vector_id)
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of stored vectors."""
        return len(self._entries)

    @property
    def dim(self) -> int:
        """Dimensionality of stored vectors (0 if empty)."""
        return self._dim

    def bucket_summary(self) -> dict[str, int]:
        """Return a mapping of bucket-key → entry-count for dashboard display.

        The bucket key is formatted as a binary string (e.g. ``"10110011"``).
        """
        return {
            "".join(str(b) for b in key): len(ids)
            for key, ids in self._buckets.items()
            if ids
        }

    def lsh_bucket_count(self) -> int:
        """Return the number of non-empty LSH buckets."""
        return sum(1 for ids in self._buckets.values() if ids)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialise all entries for vault (WAL) persistence.

        Returns
        -------
        list[dict[str, Any]]
            One JSON-serialisable dict per stored entry.
        """
        return [e.to_dict() for e in self._entries.values()]

    @classmethod
    def from_dicts(
        cls,
        records: list[dict[str, Any]],
        *,
        n_planes: int = 8,
        seed: int | None = None,
    ) -> "VectorIndex":
        """Reconstruct a :class:`VectorIndex` from serialised records.

        Parameters
        ----------
        records:
            List of dicts as produced by :meth:`to_dicts`.
        n_planes:
            Number of LSH planes for the rebuilt index.
        seed:
            Optional RNG seed.
        """
        idx = cls(n_planes=n_planes, seed=seed)
        for rec in records:
            idx.add(
                rec["vector_id"],
                rec["vector"],
                metadata=rec.get("metadata"),
            )
        return idx
