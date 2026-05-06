"""Tests for ManifoldAuth — bearer-token authentication middleware."""

from __future__ import annotations

import os
import types

import pytest

from manifold.auth import ManifoldAuth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(auth_header: str = "") -> types.SimpleNamespace:
    """Return a minimal request-like object."""
    return types.SimpleNamespace(headers={"Authorization": auth_header})


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestManifoldAuthConstructor:
    def test_explicit_secret(self):
        auth = ManifoldAuth("my-secret")
        assert auth.secret == "my-secret"

    def test_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MANIFOLD_API_KEY", "env-secret")
        auth = ManifoldAuth()
        assert auth.secret == "env-secret"

    def test_missing_secret_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MANIFOLD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MANIFOLD_API_KEY"):
            ManifoldAuth()

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MANIFOLD_API_KEY", "env-value")
        auth = ManifoldAuth("explicit-value")
        assert auth.secret == "explicit-value"


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------


class TestVerifyToken:
    def test_valid_token(self):
        auth = ManifoldAuth("correct-key")
        assert auth.verify_token("correct-key") is True

    def test_invalid_token(self):
        auth = ManifoldAuth("correct-key")
        assert auth.verify_token("wrong-key") is False

    def test_empty_token(self):
        auth = ManifoldAuth("correct-key")
        assert auth.verify_token("") is False

    def test_case_sensitive(self):
        auth = ManifoldAuth("Secret")
        assert auth.verify_token("secret") is False
        assert auth.verify_token("SECRET") is False


# ---------------------------------------------------------------------------
# is_authorized
# ---------------------------------------------------------------------------


class TestIsAuthorized:
    def test_valid_bearer_token(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("POST", "/shield", "Bearer test-key") is True

    def test_invalid_bearer_token(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("POST", "/shield", "Bearer wrong-key") is False

    def test_missing_auth_header(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("POST", "/shield", "") is False

    def test_wrong_scheme(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("POST", "/shield", "Basic test-key") is False

    def test_public_get_dashboard(self):
        auth = ManifoldAuth("test-key")
        # GET /dashboard is always public
        assert auth.is_authorized("GET", "/dashboard", "") is True

    def test_public_get_policy(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("GET", "/policy", "") is True

    def test_public_get_reputation_prefix(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("GET", "/reputation/gpt-4o", "") is True

    def test_protected_post_does_not_bypass(self):
        auth = ManifoldAuth("test-key")
        assert auth.is_authorized("POST", "/recruit", "") is False


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------


class TestMiddleware:
    def _make_handler(self) -> tuple[object, list]:
        calls: list[object] = []

        def handler(request: object) -> str:
            calls.append(request)
            return "ok"

        return handler, calls

    def test_valid_token_calls_handler(self):
        auth = ManifoldAuth("abc")
        handler, calls = self._make_handler()
        wrapped = auth.middleware(handler)
        req = _make_request("Bearer abc")
        result = wrapped(req)
        assert result == "ok"
        assert len(calls) == 1

    def test_missing_bearer_returns_401(self):
        auth = ManifoldAuth("abc")
        handler, calls = self._make_handler()
        wrapped = auth.middleware(handler)
        result = wrapped(_make_request(""))
        assert isinstance(result, dict)
        assert result["status"] == 401
        assert len(calls) == 0

    def test_missing_bearer_prefix_returns_401(self):
        auth = ManifoldAuth("abc")
        handler, calls = self._make_handler()
        wrapped = auth.middleware(handler)
        result = wrapped(_make_request("Token abc"))
        assert result["status"] == 401

    def test_wrong_token_returns_403(self):
        auth = ManifoldAuth("abc")
        handler, calls = self._make_handler()
        wrapped = auth.middleware(handler)
        result = wrapped(_make_request("Bearer wrong"))
        assert isinstance(result, dict)
        assert result["status"] == 403
        assert len(calls) == 0

    def test_response_body_is_json_string(self):
        auth = ManifoldAuth("abc")
        handler, _ = self._make_handler()
        wrapped = auth.middleware(handler)
        result = wrapped(_make_request(""))
        import json
        body = json.loads(result["body"])  # must be valid JSON
        assert "error" in body


# ---------------------------------------------------------------------------
# generate_token
# ---------------------------------------------------------------------------


class TestGenerateToken:
    def test_returns_64_char_hex(self):
        token = ManifoldAuth.generate_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_generates_unique_tokens(self):
        tokens = {ManifoldAuth.generate_token() for _ in range(20)}
        assert len(tokens) == 20

    def test_generated_token_works_as_secret(self):
        token = ManifoldAuth.generate_token()
        auth = ManifoldAuth(token)
        assert auth.verify_token(token) is True
        assert auth.verify_token("not-the-token") is False


# ---------------------------------------------------------------------------
# Fail-fast on empty / whitespace keys
# ---------------------------------------------------------------------------


class TestEmptyKeyRejection:
    def test_auth_raises_on_empty_key(self):
        """ManifoldAuth must refuse to construct with an empty key."""
        with pytest.raises(ValueError, match="MANIFOLD_API_KEY"):
            ManifoldAuth("")

    def test_auth_raises_on_whitespace_key(self):
        """ManifoldAuth must refuse to construct with a whitespace-only key."""
        with pytest.raises(ValueError, match="MANIFOLD_API_KEY"):
            ManifoldAuth("   ")

    def test_auth_raises_on_tab_key(self):
        with pytest.raises(ValueError, match="MANIFOLD_API_KEY"):
            ManifoldAuth("\t")
