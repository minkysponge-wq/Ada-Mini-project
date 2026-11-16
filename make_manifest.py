#!/usr/bin/env python3
"""Data freeze: SHA-256 of every result file plus the package versions used, so a rerun or a fresh
clone can be checked against the submitted results.

Text files (.csv, .json, .jsonl) are hashed after normalising line endings to LF, so the hashes are the
same on Windows (CRLF) and Linux/macOS (LF) and do not depend on git's line-ending settings.

Usage:
  python make_manifest.py --out out_phase1            write <out>/MANIFEST.json
  python make_manifest.py --out out_phase1 --check    compare the files on disk with MANIFEST.json
"""
import argparse, hashlib, json, os, platform, sys
from datetime import datetime, timezone
from importlib import metadata

PKGS = ["numpy", "pandas", "scipy", "networkx", "matplotlib", "huggingface_hub", "torch", "transformers"]
TEXT = (".csv", ".json", ".jsonl")


def sha(path):
    data = open(path, "rb").read()
    if path.endswith(TEXT):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def collect(out):
    files = {}
    for root, _, names in os.walk(out):
        if "preds" in root.split(os.sep):
            continue                                   # per-model predictions: hashed as one folder below
        for n in sorted(names):
            if n.endswith((".csv", ".json", ".parquet", ".jsonl")) and n != "MANIFEST.json":
                p = os.path.join(root, n)
                files[os.path.relpath(p, out).replace(os.sep, "/")] = {"sha256": sha(p)}
    preds = os.path.join(out, "phase4", "preds")
    if os.path.isdir(preds):
        h = hashlib.sha256()
        for n in sorted(os.listdir(preds)):
            h.update(n.encode()); h.update(sha(os.path.join(preds, n)).encode())
        files["phase4/preds/"] = {"sha256": h.hexdigest(), "n_files": len(os.listdir(preds))}
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_phase1")
    ap.add_argument("--check", action="store_true", help="verify files against the existing MANIFEST.json")
    a = ap.parse_args()
    path = os.path.join(a.out, "MANIFEST.json")
    files = collect(a.out)
    if a.check:
        ref = json.load(open(path, encoding="utf-8"))["files"]
        bad = [k for k in ref if files.get(k, {}).get("sha256") != ref[k]["sha256"]]
        print(f"[check] {len(ref) - len(bad)} of {len(ref)} entries match MANIFEST.json")
        for k in bad:
            print("   differs:", k)
        sys.exit(1 if bad else 0)
    vers = {}
    for p in PKGS:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    man = {"hashing": "sha256; .csv/.json/.jsonl hashed with line endings normalised to LF",
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": vers, "files": files}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2)
    print(f"[manifest] {len(files)} entries -> {a.out}/MANIFEST.json")


if __name__ == "__main__":
    main()
