"""Tests for manifold.predictor."""
import pytest
from manifold.predictor import PredictiveBrain
from manifold.brain import ManifoldBrain, BrainTask, BrainDecision


def _task(**kwargs):
    defaults = dict(prompt="test task", stakes=0.5, uncertainty=0.5, complexity=0.5)
    defaults.update(kwargs)
    return BrainTask(**defaults)


def test_original_brain_decide_still_works():
    brain = ManifoldBrain()
    decision = brain.decide(_task())
    assert isinstance(decision, BrainDecision)
    assert decision.action is not None


def test_predict_regret_bounded():
    pb = PredictiveBrain()
    for stakes in [0.0, 0.3, 0.7, 1.0]:
        for uncertainty in [0.0, 0.5, 1.0]:
            task = _task(stakes=stakes, uncertainty=uncertainty)
            regret = pb.predict_regret(task)
            assert 0.0 <= regret <= 1.0


def test_predict_and_decide_returns_brain_decision():
    pb = PredictiveBrain()
    result = pb.predict_and_decide(_task())
    assert isinstance(result, BrainDecision)
    assert result.action is not None


def test_prediction_error_logged_with_actual_outcome():
    pb = PredictiveBrain()
    pb.predict_and_decide(_task(), actual_outcome=0.3)
    assert len(pb._prediction_errors) == 1


def test_mean_prediction_error_zero_when_empty():
    pb = PredictiveBrain()
    assert pb.mean_prediction_error() == 0.0


def test_high_stakes_predicts_higher_regret():
    pb = PredictiveBrain()
    low = pb.predict_regret(_task(stakes=0.1, uncertainty=0.1, complexity=0.1))
    high = pb.predict_regret(_task(stakes=0.9, uncertainty=0.9, complexity=0.9))
    assert high > low


def test_calibration_signal_keys():
    pb = PredictiveBrain()
    signal = pb.calibration_signal()
    for key in ("mean_error", "samples", "overestimates", "underestimates"):
        assert key in signal
