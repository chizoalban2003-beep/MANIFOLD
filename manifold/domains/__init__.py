"""Domain pack loader for MANIFOLD."""
from __future__ import annotations
import importlib
from manifold.policy import ManifoldPolicy

_DOMAIN_MODULES = {
    "healthcare": "manifold.domains.healthcare",
    "finance": "manifold.domains.finance",
    "devops": "manifold.domains.devops",
    "legal": "manifold.domains.legal",
    "infrastructure": "manifold.domains.infrastructure",
    "trading": "manifold.domains.trading",
    "supply_chain": "manifold.domains.supply_chain",
}

def load_domain(name: str) -> ManifoldPolicy:
    """Load a domain policy by name. Raises ValueError for unknown names."""
    if name not in _DOMAIN_MODULES:
        known = list(_DOMAIN_MODULES.keys())
        raise ValueError(
            f"Unknown domain {name!r}. Available: {known}. "
            "Pass one of the known names or add a domain module."
        )
    module = importlib.import_module(_DOMAIN_MODULES[name])
    return module.get_policy()

def available_domains() -> list[str]:
    """Return list of domain names that can be successfully imported."""
    result = []
    for name, module_path in _DOMAIN_MODULES.items():
        try:
            importlib.import_module(module_path)
            result.append(name)
        except ImportError:
            pass
    return result
