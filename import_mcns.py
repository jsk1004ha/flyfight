#!/usr/bin/env python3
"""Compile local official MaleCNS v1.0 bulk tables for FlyFight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyfight.connectome import (
    ANNOTATIONS_FILENAME,
    EXPECTED_NEURONS,
    WEIGHTS_FILENAME,
    import_malecns,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Import official MaleCNS v1.0 Feather files into a memory-mapped CSR directory."
    )
    result.add_argument("source", type=Path, help="Directory containing the two official Feather files")
    result.add_argument("output", type=Path, help="New output directory")
    result.add_argument("--annotations", default=ANNOTATIONS_FILENAME, help="annotation filename in source")
    result.add_argument("--weights", default=WEIGHTS_FILENAME, help="connection filename in source")
    result.add_argument(
        "--allow-nonstandard-count",
        action="store_true",
        help="disable the exact 166,700-row v1.0 annotation-census check (fixtures/future releases only)",
    )
    result.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    return result


def main() -> int:
    args = parser().parse_args()
    expected = None if args.allow_nonstandard_count else EXPECTED_NEURONS
    manifest = import_malecns(
        args.source / args.annotations,
        args.source / args.weights,
        args.output,
        expected_neurons=expected,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["graph"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
