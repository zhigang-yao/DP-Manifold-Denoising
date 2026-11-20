"""Fetch exactly the upstream GPL-3.0 MF2023/MF2025 Python sources.

Third-party code is never auto-downloaded on import. Installation is an explicit
command, and every file is authenticated against its pinned Git blob SHA-1.
The upstream license is downloaded and preserved with the upstream code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from modules.paths import ROOT

UPSTREAM_REPO = "zhigang-yao/manifold-fitting"
UPSTREAM_COMMIT = "c87cf4db538c4b0567f0396903650f39339688fd"
UPSTREAM_LICENSE = "GPL-3.0 (upstream GNU GPL version 3)"
SOURCE_FILES = {
    "LICENSE": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
    "Manifold Fitting/Python/mf/manfit_ours.py": "ac94e70e44a80ad380a9c352b10f44fb6c6d317a",
    "Manifold Fitting under Unbounded Noise/Python/algo/manfit_our.py": "04cc358c5f2e56cf187586f40886887beaa67f22",
    "Manifold Fitting under Unbounded Noise/Python/algo/GD_PutM.py": "6331906d287d3628b8b90167c7ee68053f7b8a25",
    "Manifold Fitting under Unbounded Noise/Python/algo/get_obj_grad.py": "b09891e9e2ec51fa42cf9ad16ab03e76e0314bd6",
    "Manifold Fitting under Unbounded Noise/Python/utils/getUi.py": "1676acdcfb5226d91c2c7fbd587d8d9b783ca2ef",
}

def installed_root(root=None):
    return Path(root) if root is not None else ROOT / "modules" / "baselines" / "external" / "manifold-fitting"

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def file_manifest(root=None):
    base=installed_root(root)
    records={}
    for rel,expected in SOURCE_FILES.items():
        path=base/rel
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {rel}. Run: uv run python -m modules.baselines.external_mf_sources --install"
            )
        raw=path.read_bytes()
        digest=git_blob_sha(raw)
        if digest!=expected:
            raise RuntimeError(f"MF upstream file changed or corrupted: {rel} (got {digest})")
        records[rel]=dict(git_blob_sha1=digest,sha256=hashlib.sha256(raw).hexdigest(),
                          bytes=len(raw))
    return dict(repo=UPSTREAM_REPO,commit=UPSTREAM_COMMIT,
                license=UPSTREAM_LICENSE,files=records)

def install(root=None):
    """Fetch pinned raw GitHub bytes; fail closed on a hash or network mismatch."""
    base=installed_root(root)
    for rel,expected in SOURCE_FILES.items():
        target=base/rel
        if target.exists():
            if git_blob_sha(target.read_bytes())!=expected:
                raise RuntimeError(f"Existing file fails source validation: {target}")
            continue
        url=("https://raw.githubusercontent.com/" + UPSTREAM_REPO + "/" +
             UPSTREAM_COMMIT + "/" + quote(rel, safe="/"))
        with urlopen(url,timeout=45) as resp:
            raw=resp.read()
        if git_blob_sha(raw)!=expected:
            raise RuntimeError(f"Downloaded upstream source failed Git blob validation: {rel}")
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.name+".download.tmp")
        try:
            tmp.write_bytes(raw)
            os.replace(tmp,target)
        finally:
            if tmp.exists():tmp.unlink()
    provenance=file_manifest(base)
    path=base/"PINNED_SOURCE.json"
    path.write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return provenance

def main():
    p=argparse.ArgumentParser(description="Install or verify pinned GPL-3 MF upstream sources")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--install",action="store_true")
    group.add_argument("--verify",action="store_true")
    p.add_argument("--path",type=Path,help="optional local third-party root")
    args=p.parse_args()
    info=install(args.path) if args.install else file_manifest(args.path)
    print(json.dumps(dict(status="verified",repo=info["repo"],commit=info["commit"],
                          license=info["license"],files=len(info["files"])),indent=2))

if __name__=="__main__":
    main()
