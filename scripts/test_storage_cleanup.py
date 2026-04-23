"""
Smoke test: verify cleanup logic actually removes objects from Supabase Storage.

Run:
    uv run python scripts/test_storage_cleanup.py
"""

import os
import sys
from pathlib import Path

# ── Fix: project root has a supabase/ directory (migrations) that shadows the
# supabase package. sys.path[0] is '' (cwd = project root) when running a script.
# Remove both '' and the absolute project root before importing supabase.
_PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
_removed = []
for _entry in ("", _PROJECT_ROOT):
    while _entry in sys.path:
        sys.path.remove(_entry)
        _removed.append(_entry)

from supabase import create_client  # noqa: E402

for _entry in _removed:
    sys.path.insert(0, _entry)

# ── Load .env manually
_env_file = Path(_PROJECT_ROOT) / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from datetime import datetime, timezone, timedelta  # noqa: E402

SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
CROPS_BUCKET         = "ppe-crops"
SENTINEL_PATH        = "crops/_test_cleanup_sentinel.jpg"
SENTINEL_MERGE_KEY   = "_test_cleanup_sentinel"
SENTINEL_BYTES       = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"


def make_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY missing in .env")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def file_exists(client, storage_path: str) -> bool:
    folder, _, name = storage_path.rpartition("/")
    try:
        files = client.storage.from_(CROPS_BUCKET).list(folder)
        return any(f["name"] == name for f in (files or []))
    except Exception as exc:
        print(f"  [warn] list failed: {exc}")
        return False


def main() -> None:
    client = make_client()
    print("=== Storage cleanup smoke test ===\n")

    # 1. Upload sentinel
    print(f"[1] Uploading sentinel -> {SENTINEL_PATH}")
    try:
        client.storage.from_(CROPS_BUCKET).upload(
            SENTINEL_PATH, SENTINEL_BYTES,
            {"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception as exc:
        print(f"  FAIL: upload error: {exc}")
        sys.exit(1)

    public_url = client.storage.from_(CROPS_BUCKET).get_public_url(SENTINEL_PATH)
    print(f"  OK  url: {public_url}")
    print(f"  File exists before cleanup: {file_exists(client, SENTINEL_PATH)}")

    # 2. Insert fake expired row
    print("\n[2] Inserting expired ppe_labels row")
    expired_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    row = {
        "merge_key":         SENTINEL_MERGE_KEY,
        "session_id":        "test",
        "tracker_id":        "0",
        "predicted_missing": [],
        "aggregate":         "skip",
        "skipped":           True,
        "labeled_by":        "test_script",
        "labeled_at":        expired_at,
        "expire_at":         expired_at,
        "crop_url":          public_url,
        "is_auto_labeled":   False,
    }
    try:
        res = client.table("ppe_labels").upsert(row, on_conflict="merge_key").execute()
        print(f"  OK  row id: {res.data[0]['id'] if res.data else 'unknown'}")
    except Exception as exc:
        print(f"  FAIL: DB insert error: {exc}")
        client.storage.from_(CROPS_BUCKET).remove([SENTINEL_PATH])
        sys.exit(1)

    # 3. Simulate storage.remove() the same way cleanup_expired_crops() does
    print("\n[3] Calling storage.remove() (same code path as cleanup)")
    path_part = public_url.split(f"/{CROPS_BUCKET}/", 1)[-1].split("?")[0]
    print(f"  Resolved path_part: {path_part!r}")
    if path_part == public_url:
        print("  FAIL: path extraction broke — URL format unexpected")
        client.storage.from_(CROPS_BUCKET).remove([SENTINEL_PATH])
        client.table("ppe_labels").delete().eq("merge_key", SENTINEL_MERGE_KEY).execute()
        sys.exit(1)

    try:
        client.storage.from_(CROPS_BUCKET).remove([path_part])
        print("  OK  remove() returned without error")
    except Exception as exc:
        print(f"  FAIL: remove() raised: {exc}")

    # 4. Verify
    print("\n[4] Verifying file removed from Storage")
    exists_after = file_exists(client, SENTINEL_PATH)
    print(f"  File exists after remove: {exists_after}")
    print(f"\n  {'FAIL' if exists_after else 'PASS'}")

    # 5. Cleanup DB row
    print("\n[5] Removing test DB row")
    try:
        client.table("ppe_labels").delete().eq("merge_key", SENTINEL_MERGE_KEY).execute()
        print("  OK")
    except Exception as exc:
        print(f"  WARN: {exc}")

    print("\n=== Done ===")
    sys.exit(0 if not exists_after else 1)


if __name__ == "__main__":
    main()
