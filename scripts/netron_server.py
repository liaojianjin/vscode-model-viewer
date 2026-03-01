#!/usr/bin/env python3
"""Start Netron for a single model file and keep the process alive for VS Code."""

from __future__ import annotations

import argparse
import inspect
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Netron server for one model file."
    )
    parser.add_argument("--file", required=True, help="Model file path on the host machine.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", required=True, type=int, help="Bind port.")
    parser.add_argument("--log", action="store_true", help="Enable verbose Netron logging.")
    return parser.parse_args()


def start_netron(args: argparse.Namespace) -> None:
    try:
        import netron  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Python package 'netron' is not installed. Install it with: pip install netron"
        ) from error

    start_fn = netron.start
    parameters = inspect.signature(start_fn).parameters
    kwargs = {}

    if "browse" in parameters:
        kwargs["browse"] = False

    if "address" in parameters:
        kwargs["address"] = (args.host, args.port)
    else:
        if "host" in parameters:
            kwargs["host"] = args.host
        if "port" in parameters:
            kwargs["port"] = args.port

    if "log" in parameters:
        kwargs["log"] = args.log
    elif "verbose" in parameters:
        kwargs["verbose"] = args.log

    start_fn(args.file, **kwargs)


def main() -> int:
    args = parse_args()
    try:
        start_netron(args)
    except Exception as error:  # pragma: no cover - surfaced to the extension host
        print(str(error), file=sys.stderr, flush=True)
        return 1

    try:
        sys.stdin.read()
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
