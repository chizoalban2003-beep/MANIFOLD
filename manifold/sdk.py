"""manifold/sdk.py — Drop-in SDK for AI agents to integrate with MANIFOLD.

Uses only stdlib (threading, urllib.request, json). No external dependencies.

Example usage::

    sdk = ManifoldAgentSDK(
        agent_id='finance-bot',
        display_name='Finance Bot',
        capabilities=['billing', 'refund'],
        manifold_url='http://localhost:8080',
        api_key='your-key',
    )
    sdk.register()
    sdk.on_command('pause',  lambda p: my_agent.pause())
    sdk.on_command('resume', lambda p: my_agent.resume())
    sdk.start_heartbeat()   # background daemon thread
    sdk.start_polling()     # background daemon thread
    # ... later ...
    sdk.stop()
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Callable


class ManifoldAgentSDK:
    """Drop-in SDK for AI agents to integrate with MANIFOLD.

    Handles registration, heartbeat, and bidirectional command polling.
    Uses only stdlib — no external packages required.
    """

    def __init__(
        self,
        agent_id: str,
        display_name: str,
        capabilities: list[str],
        manifold_url: str,
        api_key: str,
        domain: str = "general",
    ) -> None:
        self.agent_id = agent_id
        self.display_name = display_name
        self.capabilities = capabilities
        self.manifold_url = manifold_url.rstrip("/")
        self.api_key = api_key
        self.domain = domain
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._polling_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: int = 10,
    ) -> tuple[int, dict]:
        """Low-level HTTP request using urllib. Returns (status_code, response_dict)."""
        url = f"{self.manifold_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in self._headers().items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            return exc.code, {}
        except Exception:
            return 0, {}

    # ------------------------------------------------------------------
    def register(self) -> bool:
        """POST /agents/register with agent details. Returns True on 201."""
        status, _ = self._request(
            "POST",
            "/agents/register",
            body={
                "agent_id": self.agent_id,
                "display_name": self.display_name,
                "capabilities": self.capabilities,
                "domain": self.domain,
            },
        )
        return status == 201

    # ------------------------------------------------------------------
    def heartbeat(self, status: str = "active") -> bool:
        """POST /agents/{id}/heartbeat. Returns True on 200."""
        code, _ = self._request(
            "POST",
            f"/agents/{self.agent_id}/heartbeat",
            body={"status": status},
            timeout=5,
        )
        return code == 200

    # ------------------------------------------------------------------
    def on_command(self, command: str, handler: Callable) -> None:
        """Register a callback for a specific command string."""
        self._handlers[command] = handler

    # ------------------------------------------------------------------
    def start_heartbeat(self, interval: int = 30) -> None:
        """Start a background daemon thread sending heartbeats every interval seconds."""
        self._running = True

        def _loop() -> None:
            while self._running:
                try:
                    self.heartbeat()
                except Exception:
                    pass
                time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True, name="manifold-heartbeat")
        t.start()
        self._heartbeat_thread = t

    # ------------------------------------------------------------------
    def start_polling(self) -> None:
        """Start a background daemon thread long-polling for commands."""

        def _loop() -> None:
            while self._running:
                try:
                    code, resp = self._request(
                        "GET",
                        f"/agents/{self.agent_id}/commands",
                        timeout=25,
                    )
                    if code == 200:
                        for cmd in resp.get("commands", []):
                            handler = self._handlers.get(cmd.get("command", ""))
                            if handler:
                                try:
                                    handler(cmd.get("payload", {}))
                                except Exception as exc:
                                    print(f"[ManifoldSDK] Command handler error: {exc}")
                except Exception:
                    time.sleep(5)

        t = threading.Thread(target=_loop, daemon=True, name="manifold-polling")
        t.start()
        self._polling_thread = t

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop both background threads."""
        self._running = False
