r"""Build MANIFEST.json: relative-path -> sha256 for every file in coldstart_audit.

Run from the repository root directory:
    python make_manifest.py
No absolute paths are recorded, so the manifest is location independent.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "results"}
EXCLUDE_FILES = {"MANIFEST.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            if fn in EXCLUDE_FILES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            entries[rel] = {"sha256": sha256(full), "bytes": os.path.getsize(full)}
    manifest = {"root": ".", "n_files": len(entries), "files": entries}
    out = os.path.join(ROOT, "MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(entries)} files")


if __name__ == "__main__":
    main()
