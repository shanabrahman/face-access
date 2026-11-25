"""One-time local migration of staff samples to private Supabase Storage.

Run from the project root after configuring DATABASE_URL, SUPABASE_URL, and
SUPABASE_SERVICE_ROLE_KEY. Defaults to dry-run; pass --apply to upload files.
"""
import argparse
import mimetypes
import os
from pathlib import Path

import psycopg

import storage

STAFF_DIR = Path(__file__).resolve().parent / "staff"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually upload files and update employee profile-photo paths.",
    )
    args = parser.parse_args()

    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise SystemExit("DATABASE_URL is required.")
    if not STAFF_DIR.is_dir():
        raise SystemExit(f"Staff directory not found: {STAFF_DIR}")

    plans = []
    for employee_dir in sorted(STAFF_DIR.iterdir()):
        if not employee_dir.is_dir() or employee_dir.is_symlink():
            continue
        for file_path in sorted(employee_dir.iterdir()):
            if file_path.is_file() and not file_path.is_symlink() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                object_path = f"{employee_dir.name}/{file_path.name}"
                plans.append((employee_dir.name, file_path, object_path))

    print(f"Found {len(plans)} staff sample file(s).")
    if not args.apply:
        for employee_name, file_path, object_path in plans:
            print(f"DRY RUN  {file_path} -> {storage.BUCKET_EMPLOYEE_FACES}/{object_path}")
        print("No files uploaded. Re-run with --apply to perform the migration.")
        return

    uploaded = 0
    with psycopg.connect(dsn, sslmode=os.environ.get("PGSSLMODE", "require")) as db:
        for employee_name, file_path, object_path in plans:
            employee = db.execute(
                "SELECT employee_id, profile_photo FROM employees WHERE face_name = %s",
                (employee_name,),
            ).fetchone()
            guessed_type, _ = mimetypes.guess_type(file_path.name)
            content_type = guessed_type or "application/octet-stream"
            storage.upload_bytes(
                storage.BUCKET_EMPLOYEE_FACES, object_path,
                file_path.read_bytes(), content_type, upsert=True,
            )
            # Preserve an existing profile-photo choice if it already points
            # to an object in Supabase; otherwise use the first uploaded image.
            if employee is not None:
                current_photo = employee[1] or ""
                if not current_photo or os.path.isabs(current_photo) or current_photo.startswith("staff/"):
                    if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        db.execute(
                            "UPDATE employees SET profile_photo = %s WHERE employee_id = %s",
                            (object_path, employee[0]),
                        )
            uploaded += 1
            print(f"UPLOADED {employee_name}/{file_path.name}")
    print(f"Migration complete: {uploaded} file(s) uploaded.")


if __name__ == "__main__":
    main()
