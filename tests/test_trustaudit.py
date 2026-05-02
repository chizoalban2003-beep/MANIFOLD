from manifold import run_support_trust_audit
from manifold.trustaudit import sample_support_tasks


def test_support_trust_audit_reports_regret_and_gossip() -> None:
    report = run_support_trust_audit(sample_support_tasks())

    findings = {finding.name: finding for finding in report.findings}

    assert findings["regret_reduction"].baseline_cost >= findings["regret_reduction"].manifold_cost
    assert findings["gossip_summary_notes"].baseline_cost > findings["gossip_summary_notes"].manifold_cost
    assert report.recommendations


def test_bad_tool_memory_audit_runs() -> None:
    report = run_support_trust_audit()
    bad_tool = next(finding for finding in report.findings if finding.name == "bad_tool_memory")

    assert bad_tool.baseline_cost in {0.0, 1.0}
    assert bad_tool.manifold_cost in {0.0, 1.0}
    assert bad_tool.improvement >= 0.0
