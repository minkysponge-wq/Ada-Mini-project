#!/usr/bin/env python3
"""Data freeze: SHA-256 of every input/output data file plus the package versions used,
so a rerun can be checked against the submitted results.

Usage:  python make_manifest.py --out out_phase1      (writes <out>/MANIFEST.json)
"""
import argparse, hashlib, json, os, platform, sys
from datetime import datetime, timezone
from importlib import metadata

PKGS = ["numpy", "pandas", "scipy", "networkx", "matplotlib", "huggingface_hub", "torch", "transformers"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_phase1")
    a = ap.parse_args()
    files = {}
    for root, _, names in os.walk(a.out):
        if "preds" in root.split(os.sep):
            continue                                   # per-model predictions: hashed as one folder below
        for n in sorted(names):
            if n.endswith((".csv", ".json", ".parquet")) and n != "MANIFEST.json":
                p = os.path.join(root, n)
                files[os.path.relpath(p, a.out).replace(os.sep, "/")] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    preds = os.path.join(a.out, "phase4", "preds")
    if os.path.isdir(preds):
        h = hashlib.sha256()
        for n in sorted(os.listdir(preds)):
            h.update(n.encode()); h.update(sha(os.path.join(preds, n)).encode())
        files["phase4/preds/"] = {"sha256": h.hexdigest(), "n_files": len(os.listdir(preds))}
    vers = {}
    for p in PKGS:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    man = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": vers, "files": files}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=2)
    print(f"[manifest] {len(files)} entries -> {a.out}/MANIFEST.json")


if __name__ == "__main__":
    main()
