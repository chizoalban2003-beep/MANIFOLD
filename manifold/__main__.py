"""Module entry point for ``python -m manifold``.

Supports CLI flags:
  --port PORT     HTTP listen port (default: 8080).
  --host HOST     Bind address (default: 0.0.0.0).
  --genesis       Boot as genesis root node and mint initial trust tokens.
  --daemon        Enable the ProcessWatchdog background supervisor.
  --server        Launch the HTTP server (default when --genesis/--daemon used).

Sub-commands:
  swarm-deploy <cluster.json>
      Bootstrap a multi-node MANIFOLD swarm from a cluster configuration file.

All other flags are forwarded to the original MANIFOLD CLI.
"""

from __future__ import annotations

import sys


def _is_server_mode() -> bool:
    """Return True if any server-specific flag is present in sys.argv."""
    server_flags = {"--port", "--host", "--genesis", "--daemon", "--server"}
    return any(arg in server_flags or arg.startswith("--port=") or arg.startswith("--host=") for arg in sys.argv[1:])


def _is_swarm_deploy_mode() -> bool:
    """Return True when the first positional arg is ``swarm-deploy``."""
    return len(sys.argv) >= 2 and sys.argv[1] == "swarm-deploy"


def _run_swarm_deploy(cluster_json: str) -> int:
    """Bootstrap a swarm from *cluster_json* and print a progress report."""
    import sys as _sys
    import time

    from manifold.deploy.bootstrapper import SwarmDeployer

    _RESET = "\033[0m"
    _GREEN = "\033[32m"
    _RED = "\033[31m"
    _CYAN = "\033[36m"
    _BOLD = "\033[1m"

    # Support disabling ANSI when not a TTY
    if not _sys.stdout.isatty():
        _RESET = _GREEN = _RED = _CYAN = _BOLD = ""

    def _progress(step: str, ip: str, detail: str) -> None:
        icons = {"start": "🔄", "upload": "📦", "launch": "🚀", "done": "✅"}
        icon = icons.get(step, "•")
        print(f"  {icon}  [{ip}] {detail}")
        _sys.stdout.flush()

    print(f"\n{_BOLD}MANIFOLD Swarm Deploy{_RESET}")
    print(f"  Cluster file : {_CYAN}{cluster_json}{_RESET}")

    try:
        deployer, nodes = SwarmDeployer.from_cluster_json(
            cluster_json,
            progress_callback=_progress,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\n{_RED}[ERROR] Could not load cluster file: {exc}{_RESET}")
        return 1

    if not nodes:
        print(f"\n{_RED}[ERROR] No nodes defined in cluster file.{_RESET}")
        return 1

    print(f"  Nodes        : {len(nodes)} ({nodes[0].ip} as genesis)\n")

    t_start = time.monotonic()
    results = deployer.deploy(nodes)
    elapsed = time.monotonic() - t_start

    # Summary
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print()
    print(f"{_BOLD}─── Swarm Bootstrap Summary ───{_RESET}")
    for r in results:
        status = f"{_GREEN}✓ OK{_RESET}" if r.success else f"{_RED}✗ FAILED{_RESET}"
        role = "genesis" if r.spec is nodes[0] else "worker"
        print(f"  {r.spec.ip:20s}  [{role:7s}]  {status}")
        if r.error:
            print(f"    └─ {_RED}{r.error}{_RESET}")
        if r.dashboard_url:
            print(f"    └─ Dashboard: {_CYAN}{r.dashboard_url}{_RESET}")

    print()
    if failed:
        print(f"{_RED}[WARN] {len(failed)} node(s) failed to boot.{_RESET}")
    if ok:
        genesis_result = results[0]
        if genesis_result.success:
            print(f"{_GREEN}{_BOLD}✓ Swarm Live!{_RESET}  "
                  f"Genesis dashboard → {_CYAN}{genesis_result.dashboard_url}{_RESET}")
    print(f"  Total elapsed: {elapsed:.1f}s\n")

    return 0 if not failed else 1


def main() -> int:
    if _is_swarm_deploy_mode():
        if len(sys.argv) < 3:
            print("Usage: python -m manifold swarm-deploy <cluster.json>", file=sys.stderr)
            return 1
        return _run_swarm_deploy(sys.argv[2])

    if _is_server_mode():
        import argparse

        parser = argparse.ArgumentParser(
            prog="manifold",
            description="MANIFOLD v2.0.0 — Autonomic Trust OS",
            add_help=False,
        )
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--genesis", action="store_true")
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--server", action="store_true")
        parser.add_argument("--help", "-h", action="store_true")
        args, _remaining = parser.parse_known_args()

        if args.help:
            parser.print_help()
            return 0

        if args.genesis:
            from manifold.genesis import GenesisMint, GenesisConfig

            mint = GenesisMint(GenesisConfig())
            allocs = mint.mint({"bootstrap-a": 1.0, "bootstrap-b": 2.0, "bootstrap-c": 3.0})
            if not args.daemon:
                print(f"[genesis] Minted {len(allocs)} allocations")
                for a in allocs:
                    print(f"  {a.peer_id}: {a.tokens:.2f} tokens")

        if args.daemon:
            from manifold.watchdog import ProcessWatchdog

            wd = ProcessWatchdog()
            wd.start()

        from manifold.server import run_server

        try:
            run_server(port=args.port, host=args.host)
        except KeyboardInterrupt:
            pass
        return 0

    from manifold.cli import main as cli_main

    cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
