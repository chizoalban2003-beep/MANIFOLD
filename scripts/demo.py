#!/usr/bin/env python3
"""scripts/demo.py — One command to see MANIFOLD in action.

Starts MANIFOLD with simulated agents and governance events so anyone
can experience the product without real hardware.

Usage::

    python scripts/demo.py
    python scripts/demo.py --port 9090
    python scripts/demo.py --duration 60   # run for 60 seconds then stop

Open http://localhost:8080/world in a browser while the demo runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path when run directly
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Simulated agent definitions
# ---------------------------------------------------------------------------

DEMO_AGENTS = [
    {
        "agent_id": "roomba-demo",
        "display_name": "Roomba (floor layer)",
        "capabilities": ["vacuum", "map"],
        "org_id": "demo-org",
        "domain": "home",
    },
    {
        "agent_id": "scout-drone-demo",
        "display_name": "Scout Drone (aerial layer)",
        "capabilities": ["scout", "map"],
        "org_id": "demo-org",
        "domain": "home",
    },
    {
        "agent_id": "finance-bot-demo",
        "display_name": "Finance Bot",
        "capabilities": ["billing", "analysis"],
        "org_id": "demo-org",
        "domain": "finance",
    },
    {
        "agent_id": "claude-bridge-demo",
        "display_name": "Claude Bridge",
        "capabilities": ["reasoning", "summarise"],
        "org_id": "demo-org",
        "domain": "general",
    },
]

DEMO_TASKS = [
    {"prompt": "Analyse Q3 financial report", "domain": "finance", "stakes": 0.6},
    {"prompt": "Navigate kitchen and clean", "domain": "home", "stakes": 0.4},
    {"prompt": "Scout new floor plan area", "domain": "home", "stakes": 0.5},
    {"prompt": "Summarise last week's incidents", "domain": "general", "stakes": 0.3},
    {"prompt": "Process $12,000 vendor payment", "domain": "finance", "stakes": 0.92},
    {"prompt": "Deploy code to production", "domain": "devops", "stakes": 0.8},
    {"prompt": "Review patient medication list", "domain": "healthcare", "stakes": 0.95},
    {"prompt": "Generate weekly status report", "domain": "general", "stakes": 0.2},
]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

class ManifoldDemo:
    def __init__(self, port: int = 8080, duration: int = 0) -> None:
        self.port = port
        self.duration = duration
        self.base_url = f"http://127.0.0.1:{port}"
        self.api_key = "demo-secret-key"
        self._stop_event = threading.Event()
        self._tasks_completed = 0
        self._escalations = 0
        self._refusals = 0
        self._server_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        os.environ.setdefault("MANIFOLD_API_KEY", self.api_key)
        from manifold.server import ManifoldHandler
        from http.server import HTTPServer

        server = HTTPServer(("127.0.0.1", self.port), ManifoldHandler)

        def _serve():
            while not self._stop_event.is_set():
                server.handle_request()
            server.server_close()

        self._server_thread = threading.Thread(target=_serve, daemon=True, name="manifold-demo-server")
        self._server_thread.start()

    def _wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{self.base_url}/", timeout=2)
                return True
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        return False

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        auth: bool = True,
    ) -> dict | None:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read())
            except Exception:  # noqa: BLE001
                return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_agents(self) -> None:
        for agent in DEMO_AGENTS:
            result = self._request("POST", "/agents/register", agent)
            if result and "agent_id" in result:
                print(f"  ✓ Registered: {agent['display_name']}")
            else:
                print(f"  ~ Agent registration (got: {result})")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _simulation_loop(self) -> None:
        last_status_print = time.time()
        while not self._stop_event.is_set():
            time.sleep(3)
            if self._stop_event.is_set():
                break

            # Pick a random action
            action = random.choice(["task", "heartbeat", "cell_update", "task_high_risk"])

            if action in ("task", "task_high_risk"):
                task = random.choice(DEMO_TASKS)
                if action == "task_high_risk":
                    task = {**task, "stakes": random.uniform(0.85, 0.99)}
                result = self._request("POST", "/run", task)
                if result:
                    decision = result.get("action", "?")
                    risk = result.get("risk_score", 0.0)
                    self._tasks_completed += 1
                    if decision == "escalate":
                        self._escalations += 1
                    elif decision == "refuse":
                        self._refusals += 1

            elif action == "heartbeat":
                agent = random.choice(DEMO_AGENTS)
                self._request(
                    "POST",
                    f"/agents/{agent['agent_id']}/heartbeat",
                    {"status": "active"},
                )

            elif action == "cell_update":
                # Publish a cell update via CellUpdateBus directly
                try:
                    from manifold.cell_update_bus import get_bus, CellUpdate
                    bus = get_bus()
                    bus.publish(CellUpdate(
                        x=random.randint(0, 15),
                        y=random.randint(0, 15),
                        z=0,
                        r_delta=round(random.uniform(0.1, 0.6), 2),
                        source="demo-sensor",
                        ttl=30,
                    ))
                except Exception:  # noqa: BLE001
                    pass

            # Print status every 5 seconds
            if time.time() - last_status_print >= 5.0:
                print(
                    f"  📊  tasks={self._tasks_completed}  "
                    f"escalations={self._escalations}  "
                    f"refusals={self._refusals}"
                )
                last_status_print = time.time()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("─" * 60)
        print("  MANIFOLD DEMO MODE")
        print("─" * 60)
        print(f"\n  Starting server on port {self.port} …")
        self._start_server()
        if not self._wait_ready():
            print("  ✗ Server did not become ready in time.")
            return

        print(f"  ✓ Server ready at http://127.0.0.1:{self.port}")
        print(f"\n  🌐  Open http://127.0.0.1:{self.port}/world in your browser\n")

        print("  Registering demo agents …")
        self._register_agents()

        sim_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True,
            name="manifold-demo-sim",
        )
        sim_thread.start()

        print("\n  ▶  Simulation running.  Press Ctrl+C to stop.\n")

        try:
            if self.duration > 0:
                time.sleep(self.duration)
            else:
                while not self._stop_event.is_set():
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  Shutting down …")
        finally:
            self._stop_event.set()
            sim_thread.join(timeout=5)

        print("\n─" * 60)
        print("  MANIFOLD DEMO SUMMARY")
        print(f"    Tasks completed : {self._tasks_completed}")
        print(f"    Escalations     : {self._escalations}")
        print(f"    Refusals        : {self._refusals}")
        print("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MANIFOLD Demo — one command to see MANIFOLD in action."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the MANIFOLD server (default: 8080).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = run until Ctrl+C).",
    )
    args = parser.parse_args()

    demo = ManifoldDemo(port=args.port, duration=args.duration)
    demo.run()


if __name__ == "__main__":
    main()
