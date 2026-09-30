#!/usr/bin/env python3
"""Install the experiment files onto the pinned course checkout after preflight checks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without writing files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    target = args.target.expanduser().resolve(strict=True)
    manifest = json.loads((root / "overlay_manifest.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["base_commit"]:
        raise SystemExit(f"Expected course commit {manifest['base_commit']}, got {commit}; use a matching checkout")
    copies = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Invalid manifest path: {relative}")
        source = root / "overlay" / relative
        destination = target / relative
        if not destination.resolve().is_relative_to(target):
            raise SystemExit(f"Destination escapes target: {relative}")
        if sha256(source) != entry["sha256"]:
            raise SystemExit(f"Payload checksum mismatch: {relative}")
        if destination.exists():
            checksum = sha256(destination)
            if checksum == entry["sha256"]:
                continue
            if checksum != entry["base_sha256"]:
                raise SystemExit(f"Local changes found: {relative}; preserve/review them before installing")
        copies.append((source, destination))
    # Validate every destination before the first write.
    if not args.check:
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(f"{'Validated' if args.check else 'Installed'} {len(copies)} files; existing matching files preserved")


if __name__ == "__main__":
    main()
