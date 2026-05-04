"""Module entry point for ``python -m manifold``.

Supports CLI flags:
  --port PORT     HTTP listen port (default: 8080).
  --host HOST     Bind address (default: 0.0.0.0).
  --genesis       Boot as genesis root node and mint initial trust tokens.
  --daemon        Enable the ProcessWatchdog background supervisor.
  --server        Launch the HTTP server (default when --genesis/--daemon used).

All other flags are forwarded to the original MANIFOLD CLI.
"""

from __future__ import annotations

import sys


def _is_server_mode() -> bool:
    """Return True if any server-specific flag is present in sys.argv."""
    server_flags = {"--port", "--host", "--genesis", "--daemon", "--server"}
    return any(arg in server_flags or arg.startswith("--port=") or arg.startswith("--host=") for arg in sys.argv[1:])


def main() -> int:
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
