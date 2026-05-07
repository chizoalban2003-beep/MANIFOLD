"""Tests for manifold.encoders package."""
import pytest
from manifold.encoders import encode_any
from manifold.encoders.timeseries import TimeSeriesEncoder
from manifold.encoders.structured import StructuredEncoder
from manifold.encoder_v2 import EncodedTask


def test_timeseries_spike_risk_is_one():
    enc = TimeSeriesEncoder()
    result = enc.encode([0.6, 0.7, 0.8, 0.85, 0.9, 0.93])
    assert result.risk == 1.0


def test_timeseries_stable_low_risk():
    enc = TimeSeriesEncoder()
    result = enc.encode([0.1, 0.15, 0.12, 0.11, 0.13, 0.14, 0.12, 0.11])
    assert result.risk < 0.6


def test_timeseries_empty():
    enc = TimeSeriesEncoder()
    result = enc.encode([])
    assert result.neutrality == 1.0
    assert result.risk == 0.0


def test_structured_high_fraud_score():
    enc = StructuredEncoder()
    result = enc.encode({"fraud_score": 0.9})
    assert result.risk > 0.7


def test_structured_low_fraud_high_value():
    enc = StructuredEncoder()
    result = enc.encode({"fraud_score": 0.05, "customer_value": 0.9})
    assert result.risk < 0.2


def test_all_fields_bounded_timeseries():
    enc = TimeSeriesEncoder()
    for values in [[0.5, 0.6, 0.8], [0.1] * 10, [0.99, 0.98]]:
        r = enc.encode(values)
        for field in (r.cost, r.risk, r.neutrality, r.asset):
            assert 0.0 <= field <= 1.0


def test_all_fields_bounded_structured():
    enc = StructuredEncoder()
    for row in [{"fraud_score": 0.5}, {"value": 0.8}, {"risk_score": 0.2, "revenue": 0.6}]:
        r = enc.encode(row)
        for field in (r.cost, r.risk, r.neutrality, r.asset):
            assert 0.0 <= field <= 1.0


def test_encode_any_dispatches_list():
    result = encode_any([0.5, 0.6, 0.8])
    assert isinstance(result, EncodedTask)


def test_encode_any_dispatches_dict():
    result = encode_any({"fraud_score": 0.8})
    assert isinstance(result, EncodedTask)


def test_encode_any_text_hint():
    result = encode_any("delete all data", encoder_hint="text")
    assert isinstance(result, EncodedTask)
