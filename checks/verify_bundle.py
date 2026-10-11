"""Read-only SHA-256 verification of all files listed in manifest.json."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
failed = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        failed.append(item["path"])
if failed:
    raise SystemExit("Missing or changed files: " + ", ".join(failed))
print("Verified", len(manifest["files"]), "files; all SHA-256 checksums match.")
