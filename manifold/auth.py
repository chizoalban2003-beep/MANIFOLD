"""Lightweight bearer-token authentication middleware for the MANIFOLD HTTP server.

Provides :class:`ManifoldAuth` — a zero-dependency authentication layer that
validates ``Authorization: Bearer <token>`` headers using constant-time HMAC
comparison to prevent timing attacks.

Usage
-----
::

    from manifold.auth import ManifoldAuth

    auth = ManifoldAuth(secret="my-api-key")

    # Verify a token directly
    ok = auth.verify_token("my-api-key")  # True

    # Wrap a handler function
    safe_handler = auth.middleware(my_handler)

    # Generate a new random API key
    new_key = ManifoldAuth.generate_token()

Environment variables
---------------------
``MANIFOLD_API_KEY``
    Bearer token.  Required when ``secret`` is not passed to the constructor.
``MANIFOLD_PUBLIC_ENDPOINTS``
    Comma-separated list of ``METHOD /path`` pairs that bypass authentication.
    Default: ``"GET /dashboard,GET /policy,GET /reputation"``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Callable


# Default endpoints that are always public (no auth required)
_DEFAULT_PUBLIC = frozenset(
    ep.strip()
    for ep in os.environ.get(
        "MANIFOLD_PUBLIC_ENDPOINTS",
        "GET /dashboard,GET /policy,GET /reputation",
    ).split(",")
    if ep.strip()
)


class ManifoldAuth:
    """Lightweight bearer-token authentication for the MANIFOLD HTTP server.

    Parameters
    ----------
    secret:
        The shared API key / bearer token.  Falls back to the
        ``MANIFOLD_API_KEY`` environment variable when not provided.

    Raises
    ------
    ValueError
        If neither *secret* nor ``MANIFOLD_API_KEY`` is set.

    Example
    -------
    ::

        auth = ManifoldAuth()            # reads MANIFOLD_API_KEY env var
        auth = ManifoldAuth("my-token")  # explicit secret

        # Constant-time verification
        assert auth.verify_token("my-token")

        # Generate a new random token
        new_token = ManifoldAuth.generate_token()
    """

    def __init__(self, secret: str | None = None) -> None:
        self.secret: str = secret or os.environ.get("MANIFOLD_API_KEY", "")
        if not self.secret:
            raise ValueError(
                "MANIFOLD_API_KEY environment variable must be set, "
                "or pass 'secret' to ManifoldAuth()."
            )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def verify_token(self, token: str) -> bool:
        """Constant-time comparison to prevent timing attacks.

        Parameters
        ----------
        token:
            Token extracted from the ``Authorization: Bearer <token>`` header.

        Returns
        -------
        bool
            ``True`` if *token* matches the configured secret.
        """
        return hmac.compare_digest(
            token.encode("utf-8"),
            self.secret.encode("utf-8"),
        )

    def is_authorized(self, method: str, path: str, auth_header: str) -> bool:
        """Return ``True`` if the request is authorised.

        Public endpoints (see ``MANIFOLD_PUBLIC_ENDPOINTS``) always pass.
        All other requests must carry a valid ``Authorization: Bearer`` header.

        Parameters
        ----------
        method:
            HTTP method in upper-case (e.g. ``"GET"``, ``"POST"``).
        path:
            Request path without query string (e.g. ``"/shield"``).
        auth_header:
            Value of the ``Authorization`` header (may be empty string).
        """
        # Check public endpoint exemptions
        key = f"{method} {path}"
        if key in _DEFAULT_PUBLIC:
            return True
        # Prefix match for parameterised public paths like /reputation/<id>
        for public in _DEFAULT_PUBLIC:
            pub_method, pub_path = public.split(" ", 1)
            if method == pub_method and path.startswith(pub_path):
                return True

        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header[7:]
        return self.verify_token(token)

    def middleware(self, handler: Callable) -> Callable:
        """Wrap an HTTP handler with auth checking.

        The wrapped function receives the same arguments as *handler*.
        If the ``Authorization`` header is missing or invalid, a dict
        ``{"status": 401/403, "body": '...'}`` is returned instead of
        calling the original handler.

        Parameters
        ----------
        handler:
            A callable that accepts a request-like object with a
            ``headers`` dict attribute.

        Returns
        -------
        Callable
            The auth-guarded handler.
        """

        def wrapped(request: object) -> object:
            headers = getattr(request, "headers", {})
            auth_header = headers.get("Authorization", "") if hasattr(headers, "get") else ""

            if not auth_header.startswith("Bearer "):
                return {"status": 401, "body": '{"error": "Unauthorized"}'}
            token = auth_header[7:]
            if not self.verify_token(token):
                return {"status": 403, "body": '{"error": "Forbidden"}'}
            return handler(request)

        return wrapped

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_token() -> str:
        """Generate a cryptographically secure random API key.

        Returns
        -------
        str
            A 64-character hex string derived from 32 random bytes.
        """
        return hashlib.sha256(os.urandom(32)).hexdigest()
