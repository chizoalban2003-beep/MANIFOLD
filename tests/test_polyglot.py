"""Tests for Phase 23: Polyglot Protocol — OpenAPI spec generator."""

from __future__ import annotations

import json

import pytest

from manifold.polyglot import (
    ManifoldOpenAPISpec,
    generate_openapi_spec,
    spec_to_json,
    spec_to_yaml,
)


def _canonical_paths(spec: dict) -> list[str]:
    """Return sorted list of path keys in the spec."""
    return sorted(spec["paths"].keys())


# ---------------------------------------------------------------------------
# ManifoldOpenAPISpec
# ---------------------------------------------------------------------------


class TestManifoldOpenAPISpec:
    def test_to_dict_returns_dict(self) -> None:
        spec = ManifoldOpenAPISpec()
        d = spec.to_dict()
        assert isinstance(d, dict)

    def test_openapi_version(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert d["openapi"] == "3.0.3"

    def test_info_title(self) -> None:
        d = ManifoldOpenAPISpec(title="My API").to_dict()
        assert d["info"]["title"] == "My API"

    def test_info_version(self) -> None:
        d = ManifoldOpenAPISpec(version="2.0.0").to_dict()
        assert d["info"]["version"] == "2.0.0"

    def test_server_url(self) -> None:
        d = ManifoldOpenAPISpec(server_url="https://api.example.com").to_dict()
        assert d["servers"][0]["url"] == "https://api.example.com"

    def test_has_paths(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert "paths" in d
        assert len(d["paths"]) > 0

    def test_required_paths_present(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        paths = set(d["paths"].keys())
        assert "/shield" in paths
        assert "/b2b/handshake" in paths
        assert "/reputation/{agent_id}" in paths
        assert "/recruit" in paths
        assert "/policy" in paths

    def test_has_components_schemas(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        schemas = d["components"]["schemas"]
        assert "BrainTask" in schemas
        assert "OrgPolicy" in schemas
        assert "HandshakeResult" in schemas
        assert "Error" in schemas

    def test_all_required_schemas_present(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        schemas = set(d["components"]["schemas"].keys())
        required = {
            "BrainTask",
            "InterceptResult",
            "InterceptorVeto",
            "OrgPolicy",
            "HandshakeResult",
            "ReputationScore",
            "RecruitmentRequest",
            "RecruitmentResult",
            "Error",
        }
        assert required <= schemas

    def test_has_security_schemes(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert "ManifoldHMAC" in d["components"]["securitySchemes"]

    def test_security_scheme_type(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        scheme = d["components"]["securitySchemes"]["ManifoldHMAC"]
        assert scheme["type"] == "apiKey"
        assert scheme["in"] == "header"

    def test_shield_path_post(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        shield = d["paths"]["/shield"]
        assert "post" in shield

    def test_shield_post_has_request_body(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        post = d["paths"]["/shield"]["post"]
        assert "requestBody" in post

    def test_handshake_path_post(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        hs = d["paths"]["/b2b/handshake"]
        assert "post" in hs

    def test_reputation_path_get(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        rep = d["paths"]["/reputation/{agent_id}"]
        assert "get" in rep

    def test_reputation_has_path_parameter(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        params = d["paths"]["/reputation/{agent_id}"]["get"]["parameters"]
        names = [p["name"] for p in params]
        assert "agent_id" in names

    def test_recruit_path_post(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert "post" in d["paths"]["/recruit"]

    def test_policy_path_get(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert "get" in d["paths"]["/policy"]

    def test_endpoint_ids(self) -> None:
        spec = ManifoldOpenAPISpec()
        ids = spec.endpoint_ids()
        assert "evaluateShield" in ids
        assert "b2bHandshake" in ids
        assert "getReputation" in ids
        assert "triggerRecruitment" in ids
        assert "exportPolicy" in ids

    def test_schema_names(self) -> None:
        spec = ManifoldOpenAPISpec()
        names = spec.schema_names()
        assert "BrainTask" in names
        assert "OrgPolicy" in names

    def test_tags_present(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        tag_names = {t["name"] for t in d["tags"]}
        assert "Governance" in tag_names
        assert "B2B Routing" in tag_names
        assert "Reputation" in tag_names

    def test_license_present(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        assert d["info"]["license"]["name"] == "MIT"

    def test_brain_task_schema_has_required(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        bt = d["components"]["schemas"]["BrainTask"]
        assert "required" in bt
        assert "task_id" in bt["required"]

    def test_org_policy_schema_type_object(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        op = d["components"]["schemas"]["OrgPolicy"]
        assert op["type"] == "object"

    def test_handshake_result_has_compatible(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        hr = d["components"]["schemas"]["HandshakeResult"]
        assert "compatible" in hr["properties"]

    def test_error_schema_has_code_and_message(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        err = d["components"]["schemas"]["Error"]
        assert "code" in err["properties"]
        assert "message" in err["properties"]

    def test_reputation_score_schema_has_reliability(self) -> None:
        d = ManifoldOpenAPISpec().to_dict()
        rs = d["components"]["schemas"]["ReputationScore"]
        assert "reliability" in rs["properties"]

    def test_custom_title_in_info(self) -> None:
        spec = ManifoldOpenAPISpec(title="ACME Governance API")
        d = spec.to_dict()
        assert "ACME" in d["info"]["title"]


# ---------------------------------------------------------------------------
# generate_openapi_spec
# ---------------------------------------------------------------------------


class TestGenerateOpenAPISpec:
    def test_returns_dict(self) -> None:
        d = generate_openapi_spec()
        assert isinstance(d, dict)

    def test_version_passed_through(self) -> None:
        d = generate_openapi_spec(version="9.9.9")
        assert d["info"]["version"] == "9.9.9"

    def test_server_url_passed_through(self) -> None:
        d = generate_openapi_spec(server_url="https://custom.api")
        assert d["servers"][0]["url"] == "https://custom.api"

    def test_title_passed_through(self) -> None:
        d = generate_openapi_spec(title="Custom Title")
        assert d["info"]["title"] == "Custom Title"


# ---------------------------------------------------------------------------
# spec_to_json
# ---------------------------------------------------------------------------


class TestSpecToJson:
    def test_returns_string(self) -> None:
        d = generate_openapi_spec()
        s = spec_to_json(d)
        assert isinstance(s, str)

    def test_valid_json(self) -> None:
        d = generate_openapi_spec()
        s = spec_to_json(d)
        parsed = json.loads(s)
        assert "openapi" in parsed

    def test_default_indent(self) -> None:
        d = generate_openapi_spec()
        s = spec_to_json(d)
        assert "\n" in s  # indented output has newlines

    def test_custom_indent(self) -> None:
        d = generate_openapi_spec()
        s = spec_to_json(d, indent=4)
        assert "    " in s  # 4-space indent


# ---------------------------------------------------------------------------
# spec_to_yaml
# ---------------------------------------------------------------------------


class TestSpecToYaml:
    def test_returns_string(self) -> None:
        d = generate_openapi_spec()
        y = spec_to_yaml(d)
        assert isinstance(y, str)

    def test_contains_openapi_key(self) -> None:
        d = generate_openapi_spec()
        y = spec_to_yaml(d)
        assert "openapi" in y

    def test_contains_paths_key(self) -> None:
        d = generate_openapi_spec()
        y = spec_to_yaml(d)
        assert "paths" in y

    def test_contains_shield_path(self) -> None:
        d = generate_openapi_spec()
        y = spec_to_yaml(d)
        assert "shield" in y

    def test_no_json_braces_at_top_level(self) -> None:
        # Top-level YAML should use indented block style, not JSON-like braces
        d = {"key": "value", "nested": {"a": 1}}
        y = spec_to_yaml(d)
        assert "key:" in y
        assert "nested:" in y

    def test_boolean_emitted_correctly(self) -> None:
        d = {"flag": True}
        y = spec_to_yaml(d)
        assert "true" in y

    def test_empty_dict_emitted_as_braces(self) -> None:
        d = {"empty": {}}
        y = spec_to_yaml(d)
        assert "{}" in y

    def test_empty_list_emitted_as_brackets(self) -> None:
        d = {"items": []}
        y = spec_to_yaml(d)
        assert "[]" in y
