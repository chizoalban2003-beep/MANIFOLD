"""Tests for Phase 56: Legal Proxy Bridge (bridge.py)."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest

from manifold.bridge import (
    BundleEntry,
    EvidenceBundle,
    EvidenceBundleGenerator,
    GeometricReason,
    HumanReadableManifest,
)


# ---------------------------------------------------------------------------
# BundleEntry
# ---------------------------------------------------------------------------


class TestBundleEntry:
    def test_to_dict(self) -> None:
        entry = BundleEntry(
            filename="test.json",
            content_type="application/json",
            size_bytes=42,
            sha256="abc123",
        )
        d = entry.to_dict()
        assert d["filename"] == "test.json"
        assert d["content_type"] == "application/json"
        assert d["size_bytes"] == 42
        assert d["sha256"] == "abc123"


# ---------------------------------------------------------------------------
# EvidenceBundleGenerator — basic compilation
# ---------------------------------------------------------------------------


class TestEvidenceBundleGenerator:
    def _make_gen(self) -> EvidenceBundleGenerator:
        return EvidenceBundleGenerator()

    def test_compile_empty(self) -> None:
        gen = self._make_gen()
        bundle = gen.compile("task-empty")
        assert bundle.task_id == "task-empty"
        assert isinstance(bundle.zip_bytes, bytes)
        assert len(bundle.zip_bytes) > 0
        assert bundle.bundle_sha256  # non-empty hex

    def test_compile_with_receipt(self) -> None:
        gen = self._make_gen()
        receipt = {
            "task_id": "task-001",
            "final_decision": "use_tool",
            "risk_score": 0.2,
        }
        bundle = gen.compile("task-001", receipt_dict=receipt)
        filenames = {e.filename for e in bundle.entries}
        assert "decision_receipt.json" in filenames
        assert "MANIFEST.json" in filenames

    def test_compile_with_policy(self) -> None:
        gen = self._make_gen()
        policy = {"domain": "finance", "risk_tolerance": 0.3}
        bundle = gen.compile("task-002", policy_dict=policy)
        filenames = {e.filename for e in bundle.entries}
        assert "org_policy.json" in filenames

    def test_compile_with_replay(self) -> None:
        gen = self._make_gen()
        replay = {"task_id": "task-003", "action_changed": False}
        bundle = gen.compile("task-003", replay_dict=replay)
        filenames = {e.filename for e in bundle.entries}
        assert "replay_report.json" in filenames

    def test_compile_all_artefacts(self) -> None:
        gen = self._make_gen()
        bundle = gen.compile(
            "task-full",
            receipt_dict={"task_id": "task-full"},
            policy_dict={"domain": "default"},
            replay_dict={"task_id": "task-full", "found": True},
        )
        filenames = {e.filename for e in bundle.entries}
        assert "decision_receipt.json" in filenames
        assert "org_policy.json" in filenames
        assert "replay_report.json" in filenames
        assert "MANIFEST.json" in filenames

    def test_compile_extra_files(self) -> None:
        gen = self._make_gen()
        extra = {"custom.txt": b"hello world"}
        bundle = gen.compile("task-extra", extra_files=extra)
        filenames = {e.filename for e in bundle.entries}
        assert "custom.txt" in filenames

    def test_zip_is_valid(self) -> None:
        gen = self._make_gen()
        bundle = gen.compile("task-zip", receipt_dict={"x": 1})
        buf = io.BytesIO(bundle.zip_bytes)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
        assert "decision_receipt.json" in names
        assert "MANIFEST.json" in names

    def test_zip_contents_are_valid_json(self) -> None:
        gen = self._make_gen()
        receipt = {"task_id": "t1", "val": 99}
        bundle = gen.compile("t1", receipt_dict=receipt)
        buf = io.BytesIO(bundle.zip_bytes)
        with zipfile.ZipFile(buf) as zf:
            raw = zf.read("decision_receipt.json")
        parsed = json.loads(raw)
        assert parsed["val"] == 99

    def test_bundle_sha256_is_deterministic(self) -> None:
        gen = self._make_gen()
        # Same content → same entry sha256
        b1 = gen.compile("tx", receipt_dict={"a": 1})
        b2 = gen.compile("tx", receipt_dict={"a": 1})
        # The bundle sha256 may differ (timestamps differ) but entry sha256 must match
        receipt_entry_1 = next(e for e in b1.entries if e.filename == "decision_receipt.json")
        receipt_entry_2 = next(e for e in b2.entries if e.filename == "decision_receipt.json")
        assert receipt_entry_1.sha256 == receipt_entry_2.sha256

    def test_bundle_to_dict_omits_zip_bytes(self) -> None:
        gen = self._make_gen()
        bundle = gen.compile("td", receipt_dict={"x": 1})
        d = bundle.to_dict()
        assert "zip_bytes" not in d
        assert "size_bytes" in d
        assert "entries" in d

    def test_bundle_save(self, tmp_path) -> None:
        gen = self._make_gen()
        bundle = gen.compile("ts", receipt_dict={"x": 1})
        out = tmp_path / "evidence.zip"
        bundle.save(str(out))
        assert out.exists()
        assert out.stat().st_size == len(bundle.zip_bytes)

    def test_entries_have_correct_sha256(self) -> None:
        import hashlib

        gen = self._make_gen()
        data = {"task_id": "verify-hash"}
        bundle = gen.compile("verify-hash", receipt_dict=data)
        buf = io.BytesIO(bundle.zip_bytes)
        with zipfile.ZipFile(buf) as zf:
            raw = zf.read("decision_receipt.json")
        expected = hashlib.sha256(raw).hexdigest()
        entry = next(e for e in bundle.entries if e.filename == "decision_receipt.json")
        assert entry.sha256 == expected

    def test_manifest_json_inside_zip_lists_files(self) -> None:
        gen = self._make_gen()
        bundle = gen.compile("mtest", receipt_dict={"z": 0})
        buf = io.BytesIO(bundle.zip_bytes)
        with zipfile.ZipFile(buf) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
        assert manifest["task_id"] == "mtest"
        filenames_in_manifest = [f["filename"] for f in manifest["files"]]
        assert "decision_receipt.json" in filenames_in_manifest


# ---------------------------------------------------------------------------
# GeometricReason
# ---------------------------------------------------------------------------


class TestGeometricReason:
    def _make(self, **kw) -> GeometricReason:
        defaults = dict(
            task_id="gr-001",
            action="use_tool",
            cost=0.1,
            risk=0.2,
            neutrality=0.8,
            asset=0.6,
            risk_veto_threshold=0.5,
            vetoed=False,
        )
        defaults.update(kw)
        return GeometricReason(**defaults)

    def test_grid_vector(self) -> None:
        r = self._make(cost=0.1, risk=0.2, neutrality=0.8, asset=0.6)
        assert r.grid_vector == (0.1, 0.2, 0.8, 0.6)

    def test_risk_margin_allowed(self) -> None:
        r = self._make(risk=0.2, risk_veto_threshold=0.5)
        assert abs(r.risk_margin - 0.3) < 1e-9

    def test_risk_margin_vetoed(self) -> None:
        r = self._make(risk=0.7, risk_veto_threshold=0.5, vetoed=True)
        assert r.risk_margin < 0

    def test_to_dict(self) -> None:
        r = self._make()
        d = r.to_dict()
        assert d["task_id"] == "gr-001"
        assert d["action"] == "use_tool"
        assert len(d["grid_vector"]) == 4
        assert "risk_margin" in d
        assert "vetoed" in d


# ---------------------------------------------------------------------------
# HumanReadableManifest
# ---------------------------------------------------------------------------


class TestHumanReadableManifest:
    def _gen(self) -> HumanReadableManifest:
        return HumanReadableManifest(organization_name="TestOrg")

    def _reason(self, vetoed: bool = False, risk: float = 0.2) -> GeometricReason:
        return GeometricReason(
            task_id="mr-001",
            action="use_tool",
            cost=0.1,
            risk=risk,
            neutrality=0.8,
            asset=0.6,
            risk_veto_threshold=0.5,
            vetoed=vetoed,
        )

    def test_render_allowed_contains_allowed(self) -> None:
        md = self._gen().render(self._reason(vetoed=False))
        assert "ALLOWED" in md

    def test_render_vetoed_contains_vetoed(self) -> None:
        md = self._gen().render(self._reason(vetoed=True, risk=0.8))
        assert "VETOED" in md

    def test_render_contains_task_id(self) -> None:
        md = self._gen().render(self._reason())
        assert "mr-001" in md

    def test_render_contains_org_name(self) -> None:
        md = self._gen().render(self._reason())
        assert "TestOrg" in md

    def test_render_contains_grid_table(self) -> None:
        md = self._gen().render(self._reason())
        assert "Cost" in md or "cost" in md.lower()
        assert "Risk" in md or "risk" in md.lower()

    def test_render_contains_threshold(self) -> None:
        md = self._gen().render(self._reason())
        assert "50.0%" in md  # threshold 0.5 → "50.0%"

    def test_render_notes_included(self) -> None:
        reason = GeometricReason(
            task_id="nr-001",
            action="veto",
            cost=0.0,
            risk=0.9,
            neutrality=0.5,
            asset=0.1,
            risk_veto_threshold=0.4,
            vetoed=True,
            notes="Auditor note: review manually.",
        )
        md = self._gen().render(reason)
        assert "Auditor note: review manually." in md

    def test_render_bundle_summary(self) -> None:
        gen_bundle = EvidenceBundleGenerator()
        bundle = gen_bundle.compile("bsumm", receipt_dict={"x": 1})
        md = HumanReadableManifest().render_bundle_summary(bundle)
        assert "bsumm" in md
        assert "decision_receipt.json" in md

    def test_render_is_string(self) -> None:
        md = self._gen().render(self._reason())
        assert isinstance(md, str)
        assert len(md) > 100

    def test_render_vetoed_explains_exceeded(self) -> None:
        md = self._gen().render(self._reason(vetoed=True, risk=0.8))
        # Should mention threshold exceeded
        assert "exceeded" in md.lower() or "blocked" in md.lower()

    def test_render_allowed_explains_below_threshold(self) -> None:
        md = self._gen().render(self._reason(vetoed=False, risk=0.2))
        assert "permitted" in md.lower() or "allowed" in md.lower()

    def test_render_default_policy_domain(self) -> None:
        md = self._gen().render(self._reason())
        assert "default" in md

    def test_render_custom_policy_domain(self) -> None:
        reason = GeometricReason(
            task_id="pd-001",
            action="transfer",
            cost=0.3,
            risk=0.4,
            neutrality=0.7,
            asset=0.5,
            risk_veto_threshold=0.6,
            vetoed=False,
            policy_domain="finance",
        )
        md = self._gen().render(reason)
        assert "finance" in md

    def test_plain_summary_allowed(self) -> None:
        s = HumanReadableManifest._plain_summary(self._reason(vetoed=False))
        assert "permitted" in s.lower()

    def test_plain_summary_vetoed(self) -> None:
        reason = self._reason(vetoed=True, risk=0.8)
        s = HumanReadableManifest._plain_summary(reason)
        assert "blocked" in s.lower()

    def test_interpret_risk_levels(self) -> None:
        assert "Very low" in HumanReadableManifest._interpret_risk(0.05)
        assert "Moderate" in HumanReadableManifest._interpret_risk(0.35)
        assert "High" in HumanReadableManifest._interpret_risk(0.6)
        assert "Critical" in HumanReadableManifest._interpret_risk(0.9)

    def test_interpret_cost_levels(self) -> None:
        assert "Very low" in HumanReadableManifest._interpret_cost(0.05)
        assert "Moderate" in HumanReadableManifest._interpret_cost(0.35)
        assert "High" in HumanReadableManifest._interpret_cost(0.6)
        assert "Very high" in HumanReadableManifest._interpret_cost(0.9)

    def test_interpret_neutrality_levels(self) -> None:
        assert "High" in HumanReadableManifest._interpret_neutrality(0.9)
        assert "Moderate" in HumanReadableManifest._interpret_neutrality(0.5)
        assert "Low" in HumanReadableManifest._interpret_neutrality(0.1)

    def test_interpret_asset_levels(self) -> None:
        assert "High" in HumanReadableManifest._interpret_asset(0.9)
        assert "Moderate" in HumanReadableManifest._interpret_asset(0.5)
        assert "Low" in HumanReadableManifest._interpret_asset(0.1)
