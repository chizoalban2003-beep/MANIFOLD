from __future__ import annotations
from manifold.policy import ManifoldPolicy

def get_policy() -> ManifoldPolicy:
    return ManifoldPolicy.from_template("finance")
