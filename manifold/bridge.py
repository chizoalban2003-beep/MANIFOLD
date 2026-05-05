"""Phase 56: Legal Proxy Bridge — Evidence Compilation & Audit Trail.

Provides utilities to bundle MANIFOLD decision artefacts into a portable
evidence package that can be submitted to human auditors or legal systems.

Architecture
------------
An :class:`EvidenceBundleGenerator` compiles a ``zipfile``-based archive
containing:

* The :class:`~manifold.provenance.DecisionReceipt` (as JSON).
* The signed :class:`~manifold.policy.ManifoldPolicy` snapshot (as YAML/JSON).
* A :class:`~manifold.replay.ReplayReport` (as JSON) showing the historical
  vs. current decision side-by-side.

A :class:`HumanReadableManifest` generates a Markdown summary of the
"Geometric Reason" (the 4-vector grid state) that explains why an action was
allowed or vetoed — designed for non-technical auditors.

Key classes
-----------
``BundleEntry``
    Describes one file included in an evidence bundle.
``EvidenceBundle``
    Immutable record of a compiled evidence archive.
``EvidenceBundleGenerator``
    Compiles :class:`EvidenceBundle` archives from MANIFOLD decision objects.
``GeometricReason``
    Structured explanation of a 4-vector grid decision.
``HumanReadableManifest``
    Generates Markdown audit summaries for non-technical readers.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# BundleEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleEntry:
    """Describes one file included in an evidence bundle.

    Attributes
    ----------
    filename:
        Name of the file inside the zip archive (no path separators).
    content_type:
        MIME type hint (e.g. ``"application/json"``).
    size_bytes:
        Byte size of the content.
    sha256:
        Hex-encoded SHA-256 digest of the content.
    """

    filename: str
    content_type: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceBundle:
    """Immutable record of a compiled evidence archive.

    Attributes
    ----------
    task_id:
        The task identifier this bundle relates to.
    created_at:
        POSIX timestamp when the bundle was created.
    zip_bytes:
        Raw bytes of the compiled zip archive.
    entries:
        Tuple of :class:`BundleEntry` objects describing each file in the zip.
    bundle_sha256:
        SHA-256 digest of ``zip_bytes``.
    """

    task_id: str
    created_at: float
    zip_bytes: bytes
    entries: tuple[BundleEntry, ...]
    bundle_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (zip bytes omitted)."""
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "bundle_sha256": self.bundle_sha256,
            "size_bytes": len(self.zip_bytes),
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: str) -> None:
        """Write the zip archive to *path*.

        Parameters
        ----------
        path:
            File system path for the output ``.zip`` file.
        """
        with open(path, "wb") as fh:
            fh.write(self.zip_bytes)


# ---------------------------------------------------------------------------
# EvidenceBundleGenerator
# ---------------------------------------------------------------------------


@dataclass
class EvidenceBundleGenerator:
    """Compiles MANIFOLD decision artefacts into a portable evidence bundle.

    All inputs are plain dicts / strings so that the generator has no hard
    dependency on live MANIFOLD singletons — making it easy to unit-test and
    to call from archival pipelines.

    Example
    -------
    ::

        gen = EvidenceBundleGenerator()
        bundle = gen.compile(
            task_id="task-001",
            receipt_dict=receipt.to_dict(),
            policy_dict=policy.to_dict() if policy else None,
            replay_dict=report.to_dict() if report else None,
        )
        bundle.save("evidence_task_001.zip")
    """

    def compile(
        self,
        task_id: str,
        *,
        receipt_dict: dict[str, Any] | None = None,
        policy_dict: dict[str, Any] | None = None,
        replay_dict: dict[str, Any] | None = None,
        extra_files: dict[str, bytes] | None = None,
    ) -> EvidenceBundle:
        """Compile an :class:`EvidenceBundle` for *task_id*.

        Parameters
        ----------
        task_id:
            Identifier for the decision being archived.
        receipt_dict:
            JSON-serialisable dict from
            :meth:`~manifold.provenance.DecisionReceipt.to_dict`.
        policy_dict:
            JSON-serialisable dict from
            :meth:`~manifold.policy.ManifoldPolicy.to_dict`.
        replay_dict:
            JSON-serialisable dict from
            :meth:`~manifold.replay.ReplayReport.to_dict`.
        extra_files:
            Additional ``{filename: bytes}`` entries to include verbatim.

        Returns
        -------
        EvidenceBundle
            Archive containing all provided artefacts plus a manifest.
        """
        import hashlib

        created_at = time.time()
        entries: list[BundleEntry] = []
        file_contents: dict[str, bytes] = {}

        def _add_json(filename: str, data: dict[str, Any]) -> None:
            raw = json.dumps(data, indent=2, default=str).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            entries.append(
                BundleEntry(
                    filename=filename,
                    content_type="application/json",
                    size_bytes=len(raw),
                    sha256=digest,
                )
            )
            file_contents[filename] = raw

        if receipt_dict is not None:
            _add_json("decision_receipt.json", receipt_dict)

        if policy_dict is not None:
            _add_json("org_policy.json", policy_dict)

        if replay_dict is not None:
            _add_json("replay_report.json", replay_dict)

        if extra_files:
            for fname, fbytes in extra_files.items():
                digest = hashlib.sha256(fbytes).hexdigest()
                entries.append(
                    BundleEntry(
                        filename=fname,
                        content_type="application/octet-stream",
                        size_bytes=len(fbytes),
                        sha256=digest,
                    )
                )
                file_contents[fname] = fbytes

        # Build the bundle manifest
        manifest = {
            "task_id": task_id,
            "created_at": created_at,
            "generator": "MANIFOLD EvidenceBundleGenerator v56",
            "files": [e.to_dict() for e in entries],
        }
        manifest_raw = json.dumps(manifest, indent=2, default=str).encode("utf-8")
        manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
        entries.append(
            BundleEntry(
                filename="MANIFEST.json",
                content_type="application/json",
                size_bytes=len(manifest_raw),
                sha256=manifest_digest,
            )
        )
        file_contents["MANIFEST.json"] = manifest_raw

        # Pack into a zip archive in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, fbytes in file_contents.items():
                zf.writestr(fname, fbytes)
        zip_bytes = buf.getvalue()
        bundle_sha256 = hashlib.sha256(zip_bytes).hexdigest()

        return EvidenceBundle(
            task_id=task_id,
            created_at=created_at,
            zip_bytes=zip_bytes,
            entries=tuple(entries),
            bundle_sha256=bundle_sha256,
        )


# ---------------------------------------------------------------------------
# GeometricReason
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometricReason:
    """Structured explanation of a 4-vector grid decision.

    The MANIFOLD decision grid operates on four dimensions:
    ``[cost, risk, neutrality, asset]``.  This class captures those values
    and the resulting veto/allow decision in a form that can be rendered
    as human-readable Markdown.

    Attributes
    ----------
    task_id:
        Identifier for the decision.
    action:
        The final action taken (e.g. ``"use_tool"``, ``"veto"``).
    cost:
        Cost component of the grid vector (0–1).
    risk:
        Risk component of the grid vector (0–1).
    neutrality:
        Neutrality component of the grid vector (0–1).
    asset:
        Asset/value component of the grid vector (0–1).
    risk_veto_threshold:
        The threshold that triggered or nearly triggered a veto.
    vetoed:
        ``True`` if the action was vetoed by the interceptor.
    policy_domain:
        The active policy domain name (e.g. ``"finance"``).
    notes:
        Free-form audit notes.
    """

    task_id: str
    action: str
    cost: float
    risk: float
    neutrality: float
    asset: float
    risk_veto_threshold: float
    vetoed: bool
    policy_domain: str = "default"
    notes: str = ""

    @property
    def grid_vector(self) -> tuple[float, float, float, float]:
        """Return the 4-vector as ``(cost, risk, neutrality, asset)``."""
        return (self.cost, self.risk, self.neutrality, self.asset)

    @property
    def risk_margin(self) -> float:
        """Distance between risk and the veto threshold (negative = vetoed)."""
        return self.risk_veto_threshold - self.risk

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "task_id": self.task_id,
            "action": self.action,
            "grid_vector": list(self.grid_vector),
            "risk_veto_threshold": round(self.risk_veto_threshold, 4),
            "vetoed": self.vetoed,
            "risk_margin": round(self.risk_margin, 4),
            "policy_domain": self.policy_domain,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# HumanReadableManifest
# ---------------------------------------------------------------------------


@dataclass
class HumanReadableManifest:
    """Generates Markdown audit summaries for non-technical auditors.

    The summary explains *why* a MANIFOLD action was allowed or vetoed in
    plain language, making the 4-vector grid geometry understandable to
    compliance officers and legal teams.

    Example
    -------
    ::

        manifest_gen = HumanReadableManifest()
        reason = GeometricReason(
            task_id="task-001",
            action="use_tool",
            cost=0.2, risk=0.15, neutrality=0.8, asset=0.6,
            risk_veto_threshold=0.5,
            vetoed=False,
        )
        md = manifest_gen.render(reason)
        print(md)
    """

    organization_name: str = "MANIFOLD"

    def render(self, reason: GeometricReason) -> str:
        """Render a Markdown audit summary for *reason*.

        Parameters
        ----------
        reason:
            A :class:`GeometricReason` describing the decision.

        Returns
        -------
        str
            Multi-line Markdown string suitable for non-technical readers.
        """
        cost, risk, neutrality, asset = reason.grid_vector
        verdict = "**VETOED ❌**" if reason.vetoed else "**ALLOWED ✅**"
        risk_pct = f"{risk * 100:.1f}%"
        threshold_pct = f"{reason.risk_veto_threshold * 100:.1f}%"
        margin_pct = f"{abs(reason.risk_margin) * 100:.1f}%"
        margin_direction = "below" if reason.risk_margin >= 0 else "above"

        lines = [
            f"# MANIFOLD Decision Audit — Task `{reason.task_id}`",
            "",
            f"**Organisation:** {self.organization_name}  ",
            f"**Policy Domain:** `{reason.policy_domain}`  ",
            f"**Final Decision:** {verdict}  ",
            f"**Action Requested:** `{reason.action}`  ",
            "",
            "---",
            "",
            "## Geometric Reason (4-Vector Grid)",
            "",
            "The MANIFOLD Trust OS evaluates every action against a four-dimensional",
            "risk geometry.  Each dimension is scored in the range **0.0 – 1.0**.",
            "",
            "| Dimension | Score | Interpretation |",
            "|-----------|-------|----------------|",
            f"| 💰 Cost       | `{cost:.3f}` | {self._interpret_cost(cost)} |",
            f"| ⚠️  Risk       | `{risk:.3f}` | {self._interpret_risk(risk)} |",
            f"| ⚖️  Neutrality | `{neutrality:.3f}` | {self._interpret_neutrality(neutrality)} |",
            f"| 🏆 Asset      | `{asset:.3f}` | {self._interpret_asset(asset)} |",
            "",
            "---",
            "",
            "## Veto Analysis",
            "",
            f"The active **Risk Veto Threshold** for the `{reason.policy_domain}` policy",
            f"domain is **{threshold_pct}**.",
            "",
        ]

        if reason.vetoed:
            lines += [
                f"The submitted action's risk score of **{risk_pct}** exceeded the",
                f"veto threshold by **{margin_pct}**.  MANIFOLD automatically blocked",
                "the action to protect the system from unsafe outcomes.",
                "",
            ]
        else:
            lines += [
                f"The submitted action's risk score of **{risk_pct}** is **{margin_pct}**",
                f"{margin_direction} the veto threshold.  The action was permitted to proceed.",
                "",
            ]

        lines += [
            "---",
            "",
            "## Plain-Language Summary",
            "",
            self._plain_summary(reason),
            "",
        ]

        if reason.notes:
            lines += [
                "---",
                "",
                "## Auditor Notes",
                "",
                reason.notes,
                "",
            ]

        lines += [
            "---",
            "",
            "*This document was automatically generated by the MANIFOLD Legal Proxy Bridge.*",
            f"*Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}*",
        ]

        return "\n".join(lines)

    def render_bundle_summary(self, bundle: EvidenceBundle) -> str:
        """Render a Markdown index for an :class:`EvidenceBundle`.

        Parameters
        ----------
        bundle:
            The evidence bundle to summarise.

        Returns
        -------
        str
            Markdown string listing the bundle contents.
        """
        lines = [
            f"# Evidence Bundle — Task `{bundle.task_id}`",
            "",
            f"**Bundle SHA-256:** `{bundle.bundle_sha256}`  ",
            f"**Created:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(bundle.created_at))}  ",
            f"**Total size:** {len(bundle.zip_bytes):,} bytes  ",
            "",
            "## Included Files",
            "",
            "| File | Type | Size | SHA-256 (prefix) |",
            "|------|------|------|------------------|",
        ]
        for entry in bundle.entries:
            prefix = entry.sha256[:16]
            lines.append(
                f"| `{entry.filename}` | `{entry.content_type}` "
                f"| {entry.size_bytes:,} B | `{prefix}…` |"
            )

        lines += [
            "",
            "---",
            "",
            "*Generated by MANIFOLD Legal Proxy Bridge (Phase 56).*",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private interpretation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_cost(v: float) -> str:
        if v < 0.2:
            return "Very low cost — minimal resource consumption."
        if v < 0.5:
            return "Moderate cost — within normal operating bounds."
        if v < 0.75:
            return "High cost — significant resources required."
        return "Very high cost — potential budget overrun risk."

    @staticmethod
    def _interpret_risk(v: float) -> str:
        if v < 0.2:
            return "Very low risk — safe to proceed."
        if v < 0.5:
            return "Moderate risk — standard caution advised."
        if v < 0.75:
            return "High risk — elevated oversight required."
        return "Critical risk — action flagged for immediate review."

    @staticmethod
    def _interpret_neutrality(v: float) -> str:
        if v >= 0.7:
            return "High neutrality — action is well-balanced."
        if v >= 0.4:
            return "Moderate neutrality — minor bias detected."
        return "Low neutrality — action may be one-sided."

    @staticmethod
    def _interpret_asset(v: float) -> str:
        if v >= 0.7:
            return "High value — significant positive return expected."
        if v >= 0.4:
            return "Moderate value — reasonable expected benefit."
        return "Low value — marginal expected return."

    @staticmethod
    def _plain_summary(reason: GeometricReason) -> str:
        if reason.vetoed:
            return (
                f"MANIFOLD blocked the `{reason.action}` action because its "
                f"risk score ({reason.risk:.3f}) exceeded the safety threshold "
                f"({reason.risk_veto_threshold:.3f}) configured for the "
                f"`{reason.policy_domain}` policy domain.  This is a safety-"
                "critical interlock designed to prevent harmful autonomous actions."
            )
        return (
            f"MANIFOLD permitted the `{reason.action}` action.  The risk score "
            f"({reason.risk:.3f}) is within the acceptable range defined by the "
            f"`{reason.policy_domain}` policy (threshold: "
            f"{reason.risk_veto_threshold:.3f}).  All four vector dimensions — "
            "cost, risk, neutrality, and asset value — were evaluated and found "
            "to be within policy limits."
        )
