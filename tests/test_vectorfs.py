"""Tests for Phase 47: Native Vector File System (manifold/vectorfs.py)."""

from __future__ import annotations

import math
import random

import pytest

from manifold.vectorfs import (
    LSHIndex,
    SearchResult,
    VectorEntry,
    VectorIndex,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# cosine_similarity — unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_are_zero(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_are_minus_one(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_zero_vector_a_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b_returns_zero(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_return_zero(self) -> None:
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self) -> None:
        a = [1.0, 0.0]
        b = [math.cos(math.pi / 4), math.sin(math.pi / 4)]
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(math.cos(math.pi / 4), abs=1e-9)

    def test_high_dimensional(self) -> None:
        rng = random.Random(42)
        v = [rng.gauss(0, 1) for _ in range(512)]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_scale_invariant(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # same direction, doubled magnitude
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_single_element_vectors(self) -> None:
        assert cosine_similarity([5.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([5.0], [-5.0]) == pytest.approx(-1.0)
        assert cosine_similarity([0.0], [5.0]) == 0.0


# ---------------------------------------------------------------------------
# VectorEntry
# ---------------------------------------------------------------------------


class TestVectorEntry:
    def test_to_dict_round_trip(self) -> None:
        e = VectorEntry(vector_id="v1", vector=[1.0, 2.0], metadata={"label": "a"})
        d = e.to_dict()
        assert d["vector_id"] == "v1"
        assert d["vector"] == [1.0, 2.0]
        assert d["metadata"] == {"label": "a"}

    def test_default_metadata_is_empty(self) -> None:
        e = VectorEntry(vector_id="v2", vector=[0.5])
        assert e.metadata == {}

    def test_vector_is_mutable_list(self) -> None:
        e = VectorEntry(vector_id="v3", vector=[1.0, 2.0])
        assert isinstance(e.vector, list)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_frozen(self) -> None:
        r = SearchResult(vector_id="v1", similarity=0.95)
        with pytest.raises((AttributeError, TypeError)):
            r.vector_id = "v2"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        r = SearchResult(vector_id="x", similarity=0.7)
        d = r.to_dict()
        assert d["vector_id"] == "x"
        assert d["similarity"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# LSHIndex
# ---------------------------------------------------------------------------


class TestLSHIndex:
    def test_hash_is_binary_tuple(self) -> None:
        lsh = LSHIndex(n_planes=4, seed=0)
        lsh.fit(dim=3)
        h = lsh.hash_vector([1.0, 0.5, -0.3])
        assert len(h) == 4
        assert all(b in (0, 1) for b in h)

    def test_same_vector_same_hash(self) -> None:
        lsh = LSHIndex(n_planes=8, seed=1)
        lsh.fit(dim=4)
        v = [0.1, 0.2, 0.3, 0.4]
        assert lsh.hash_vector(v) == lsh.hash_vector(v)

    def test_not_fitted_raises(self) -> None:
        lsh = LSHIndex(n_planes=4)
        with pytest.raises(RuntimeError, match="not been fitted"):
            lsh.hash_vector([1.0, 2.0])

    def test_dimension_mismatch_raises(self) -> None:
        lsh = LSHIndex(n_planes=4, seed=0)
        lsh.fit(dim=3)
        with pytest.raises(ValueError, match="Expected 3-dim"):
            lsh.hash_vector([1.0, 2.0])

    def test_dim_property(self) -> None:
        lsh = LSHIndex(n_planes=4, seed=0)
        assert lsh.dim == 0
        lsh.fit(dim=5)
        assert lsh.dim == 5

    def test_deterministic_with_seed(self) -> None:
        lsh1 = LSHIndex(n_planes=8, seed=99)
        lsh1.fit(dim=4)
        lsh2 = LSHIndex(n_planes=8, seed=99)
        lsh2.fit(dim=4)
        v = [1.0, -1.0, 0.5, 0.5]
        assert lsh1.hash_vector(v) == lsh2.hash_vector(v)

    def test_random_data_partitioned_into_buckets(self) -> None:
        """LSH should spread 200 random vectors across multiple buckets."""
        lsh = LSHIndex(n_planes=8, seed=42)
        lsh.fit(dim=16)
        rng = random.Random(42)
        buckets: set[tuple[int, ...]] = set()
        for _ in range(200):
            v = [rng.gauss(0, 1) for _ in range(16)]
            buckets.add(lsh.hash_vector(v))
        # With 8 planes there are 256 possible buckets; 200 vectors should
        # land in several different buckets.
        assert len(buckets) > 1


# ---------------------------------------------------------------------------
# VectorIndex — add / len / dim
# ---------------------------------------------------------------------------


class TestVectorIndexAdd:
    def test_add_and_len(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("a", [1.0, 0.0])
        idx.add("b", [0.0, 1.0])
        assert len(idx) == 2

    def test_dim_is_set_on_first_add(self) -> None:
        idx = VectorIndex(seed=0)
        assert idx.dim == 0
        idx.add("v", [1.0, 2.0, 3.0])
        assert idx.dim == 3

    def test_zero_length_vector_raises(self) -> None:
        idx = VectorIndex()
        with pytest.raises(ValueError):
            idx.add("empty", [])

    def test_dimension_mismatch_raises(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("v1", [1.0, 2.0])
        with pytest.raises(ValueError, match="Expected 2-dim"):
            idx.add("v2", [1.0, 2.0, 3.0])

    def test_update_existing_id(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("v", [1.0, 0.0])
        idx.add("v", [0.0, 1.0])
        assert len(idx) == 1  # overwrite, not duplicate
        assert idx.get("v") is not None
        assert idx.get("v").vector == [0.0, 1.0]  # type: ignore[union-attr]

    def test_metadata_stored(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("doc1", [1.0, 0.0], metadata={"text": "hello"})
        entry = idx.get("doc1")
        assert entry is not None
        assert entry.metadata["text"] == "hello"

    def test_metadata_is_copied(self) -> None:
        meta = {"key": "val"}
        idx = VectorIndex(seed=0)
        idx.add("doc1", [1.0, 0.0], metadata=meta)
        meta["key"] = "changed"
        assert idx.get("doc1").metadata["key"] == "val"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# VectorIndex — remove
# ---------------------------------------------------------------------------


class TestVectorIndexRemove:
    def test_remove_existing(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("a", [1.0, 0.0])
        assert idx.remove("a") is True
        assert len(idx) == 0

    def test_remove_nonexistent_returns_false(self) -> None:
        idx = VectorIndex(seed=0)
        assert idx.remove("ghost") is False

    def test_get_after_remove_is_none(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("a", [1.0, 0.0])
        idx.remove("a")
        assert idx.get("a") is None


# ---------------------------------------------------------------------------
# VectorIndex — search
# ---------------------------------------------------------------------------


class TestVectorIndexSearch:
    def test_empty_index_returns_empty(self) -> None:
        idx = VectorIndex(seed=0)
        assert idx.search([1.0, 0.0]) == []

    def test_finds_closest_vector(self) -> None:
        idx = VectorIndex(n_planes=4, seed=7)
        idx.add("close", [0.9, 0.1])
        idx.add("far", [0.0, 1.0])
        results = idx.search([1.0, 0.0], k=2)
        assert len(results) >= 1
        # close should be ranked first
        assert results[0].vector_id == "close"

    def test_k_limits_results(self) -> None:
        idx = VectorIndex(n_planes=4, seed=0)
        for i in range(10):
            idx.add(f"v{i}", [float(i), 0.0])
        results = idx.search([5.0, 0.0], k=3)
        assert len(results) <= 3

    def test_results_sorted_descending(self) -> None:
        idx = VectorIndex(n_planes=8, seed=0)
        idx.add("a", [1.0, 0.0])
        idx.add("b", [0.7, 0.3])
        idx.add("c", [0.0, 1.0])
        results = idx.search([1.0, 0.0], k=3)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_dimension_mismatch_raises(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("v", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Expected 3-dim"):
            idx.search([1.0, 2.0])

    def test_fallback_to_full_scan_when_bucket_empty(self) -> None:
        """If the query's LSH bucket is empty, search must still return results."""
        # Use n_planes=16 so buckets are very granular
        idx = VectorIndex(n_planes=16, seed=99)
        idx.add("only", [1.0, 0.0, 0.0])
        # The query is in a different direction — may land in a different bucket
        results = idx.search([0.0, 1.0, 0.0], k=1)
        # Should still return the one available entry (via fallback scan)
        assert len(results) == 1
        assert results[0].vector_id == "only"


# ---------------------------------------------------------------------------
# VectorIndex — bucket_summary / lsh_bucket_count
# ---------------------------------------------------------------------------


class TestVectorIndexBuckets:
    def test_bucket_summary_keys_are_binary_strings(self) -> None:
        idx = VectorIndex(n_planes=4, seed=0)
        idx.add("v1", [1.0, 0.0])
        summary = idx.bucket_summary()
        for key in summary:
            assert all(c in "01" for c in key)

    def test_bucket_summary_values_are_counts(self) -> None:
        idx = VectorIndex(n_planes=4, seed=0)
        idx.add("v1", [1.0, 0.0])
        idx.add("v2", [0.9, 0.1])
        summary = idx.bucket_summary()
        assert sum(summary.values()) == 2

    def test_lsh_bucket_count_positive(self) -> None:
        idx = VectorIndex(n_planes=4, seed=0)
        for i in range(5):
            idx.add(f"v{i}", [float(i), float(i + 1)])
        assert idx.lsh_bucket_count() >= 1

    def test_random_data_uses_multiple_buckets(self) -> None:
        """200 random 32-dim vectors should land in >1 LSH bucket."""
        rng = random.Random(1)
        idx = VectorIndex(n_planes=8, seed=1)
        for i in range(200):
            v = [rng.gauss(0, 1) for _ in range(32)]
            idx.add(f"v{i}", v)
        assert idx.lsh_bucket_count() > 1


# ---------------------------------------------------------------------------
# VectorIndex — serialisation / from_dicts round-trip
# ---------------------------------------------------------------------------


class TestVectorIndexSerialisation:
    def test_to_dicts_empty(self) -> None:
        idx = VectorIndex()
        assert idx.to_dicts() == []

    def test_to_dicts_fields(self) -> None:
        idx = VectorIndex(seed=0)
        idx.add("v1", [1.0, 2.0], metadata={"label": "test"})
        records = idx.to_dicts()
        assert len(records) == 1
        assert records[0]["vector_id"] == "v1"
        assert records[0]["vector"] == [1.0, 2.0]
        assert records[0]["metadata"] == {"label": "test"}

    def test_from_dicts_round_trip(self) -> None:
        original = VectorIndex(seed=42)
        original.add("a", [1.0, 0.0], metadata={"tag": "alpha"})
        original.add("b", [0.0, 1.0], metadata={"tag": "beta"})

        records = original.to_dicts()
        rebuilt = VectorIndex.from_dicts(records, n_planes=8, seed=42)

        assert len(rebuilt) == 2
        assert rebuilt.get("a") is not None
        assert rebuilt.get("b") is not None
        assert rebuilt.get("a").metadata["tag"] == "alpha"  # type: ignore[union-attr]

    def test_from_dicts_empty(self) -> None:
        idx = VectorIndex.from_dicts([])
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# Integration: LSH correctly partitions random data into buckets
# ---------------------------------------------------------------------------


class TestLSHPartitioning:
    def test_nearby_vectors_tend_to_share_bucket(self) -> None:
        """Vectors that are very close should share an LSH bucket more often
        than randomly expected."""
        rng = random.Random(10)
        dim = 32
        n_planes = 8
        idx = VectorIndex(n_planes=n_planes, seed=10)

        base = [rng.gauss(0, 1) for _ in range(dim)]
        # Add noisy near-duplicate
        near = [b + rng.gauss(0, 0.05) for b in base]
        # Add a far vector (opposite direction)
        far = [-b for b in base]

        idx.add("base", base)
        idx.add("near", near)
        idx.add("far", far)

        # The base and near should both appear in a search for the base vector
        results = idx.search(base, k=3)
        result_ids = {r.vector_id for r in results}
        assert "base" in result_ids
        # near should score higher than far
        sim_near = cosine_similarity(base, near)
        sim_far = cosine_similarity(base, far)
        assert sim_near > sim_far

    def test_large_index_search_returns_results(self) -> None:
        rng = random.Random(5)
        idx = VectorIndex(n_planes=8, seed=5)
        dim = 16
        for i in range(100):
            v = [rng.gauss(0, 1) for _ in range(dim)]
            idx.add(f"v{i}", v)

        query = [rng.gauss(0, 1) for _ in range(dim)]
        results = idx.search(query, k=5)
        assert 1 <= len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
