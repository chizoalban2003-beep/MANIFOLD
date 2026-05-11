"""ManifoldPipeline — integrates all 6 brain-system modules into one pipeline."""
from __future__ import annotations

from manifold.encoders import encode_any
from manifold.workspace import GlobalWorkspace
from manifold.predictor import PredictiveBrain
from manifold.cognitive_map import CognitiveMap
from manifold.cooccurrence import ToolCooccurrenceGraph
from manifold.consolidator import MemoryConsolidator
from manifold.brain import ManifoldBrain, BrainTask
from manifold.gridmapper import CellVector
from manifold.policy_rules import PolicyRuleEngine


class ManifoldPipeline:
    """Wires all 6 brain-system modules into one coherent pipeline."""

    def __init__(self, brain: ManifoldBrain | None = None) -> None:
        if brain is None:
            brain = ManifoldBrain()
        self._brain = brain
        self._encoder_workspace = GlobalWorkspace()
        self._predictor = PredictiveBrain(brain=self._brain)
        self._cognitive_map = CognitiveMap()
        self._cooccurrence = ToolCooccurrenceGraph()
        self._consolidator = MemoryConsolidator()
        self._rule_engine = PolicyRuleEngine()

    # ------------------------------------------------------------------
    def run(
        self,
        prompt: str,
        tools_used: list[str] | None = None,
        data: object = None,
        encoder_hint: str = "auto",
        explicit_domain: str | None = None,
        stakes: float = 0.5,
        uncertainty: float = 0.5,
        org_id: str = "",
    ) -> dict:
        """Execute the full 6-step pipeline and return a result dict."""

        # Step 1: encode
        encoded = encode_any(data if data is not None else prompt, encoder_hint)

        # Step 2: route
        domain = self._encoder_workspace.route_task(prompt, explicit_domain)
        # Override stakes with encoded risk when structured/timeseries data provided
        effective_stakes = encoded.risk if data is not None else stakes

        # Step 2b: policy rule engine — short-circuit if a rule matches
        rule_context = {
            "domain": domain,
            "stakes": effective_stakes,
            "risk_score": encoded.risk,
            "prompt": prompt,
            "org_id": org_id,
            "tools_used": tools_used or [],
        }
        rule_action = self._rule_engine.evaluate(rule_context)
        if rule_action is not None:
            return {
                "action": rule_action,
                "domain": domain,
                "risk_score": encoded.risk,
                "encoded": encoded,
                "nearest_cells": [],
                "flagged_tools": [],
                "rule_applied": True,
            }

        # Step 3: decide
        task = BrainTask(
            prompt=prompt,
            domain=domain,
            stakes=effective_stakes,
            uncertainty=uncertainty,
        )
        decision = self._predictor.predict_and_decide(task)

        # Step 4: navigate
        world = self._brain.map_task_to_world(task)
        query_vector = CellVector(
            cost=encoded.cost,
            risk=encoded.risk,
            neutrality=encoded.neutrality,
            asset=encoded.asset,
        )
        nearest = self._cognitive_map.navigate(query_vector, world, k=3)

        # Step 5: check_tools
        flagged_tools: list[str] = []
        if tools_used:
            success = decision.action not in ("refuse", "stop")
            self._cooccurrence.record_task(tools_used, success)
            flagged_tools = self._cooccurrence.propagate_flag(tools_used[0]) if tools_used else []

        return {
            "action": decision.action,
            "domain": domain,
            "risk_score": decision.risk_score,
            "encoded": encoded,
            "nearest_cells": [
                {"row": r["row"], "col": r["col"], "distance": r["distance"]}
                for r in nearest
            ],
            "flagged_tools": flagged_tools,
            "rule_applied": False,
        }

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        row: int,
        col: int,
        action: str,
        success: bool,
        risk_score: float,
    ) -> None:
        """Delegate to CognitiveMap."""
        self._cognitive_map.record_outcome(row, col, action, success, risk_score)

    # ------------------------------------------------------------------
    def nightly_consolidation(self, outcome_log: list[dict]) -> list:
        """Delegate to MemoryConsolidator."""
        return self._consolidator.consolidate(outcome_log)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return pipeline-level statistics."""
        return {
            "predictor_mean_error": self._predictor.mean_prediction_error(),
            "cooccurrence": self._cooccurrence.summary(),
            "promoted_rule_count": len(self._consolidator.promoted_rules()),
        }
