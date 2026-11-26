"""SQLite persistence for employee profiles and attendance."""
import csv
import hashlib
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time as datetime_time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATABASE_FILE = os.path.join(os.path.dirname(__file__), "attendance.db")
LEGACY_ATTENDANCE_CLEAR_EVENT_ID_SETTING = "legacy_attendance_cleared_through_event_id"
DEFAULT_SETTINGS = {
    "timezone": "Asia/Qatar",
    "office_start": "09:00",
    "office_end": "18:00",
    "grace_period_minutes": "15",
    "standard_work_hours": "8",
    "face_threshold": "0.40",
    "working_days": '["sun","mon","tue","wed","thu"]',
    "holidays": "[]",
    "attendance_correction_period_days": "30",
}

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
ATTENDANCE_STATUSES = {
    "ON_TIME",
    "GRACE_PERIOD",
    "LATE",
    "ABSENT",
    "CURRENTLY_WORKING",
    "CHECKED_OUT",
    "ON_LEAVE",
    "WEEKLY_OFF",
    "HOLIDAY",
    "NOT_CHECKED_IN",
    "NOT_SCHEDULED",
}


def classify_attendance(record):
    """Derive punctuality and lateness from the original check-in and current shift."""
    check_in = datetime.fromisoformat(record["check_in"])
    shift_start = datetime_time.fromisoformat(
        record.get("shift_start") or record.get("start_time")
    )
    scheduled_start = datetime.combine(
        date.fromisoformat(record["attendance_date"]), shift_start, check_in.tzinfo
    )
    elapsed_seconds = max(0, (check_in - scheduled_start).total_seconds())
    minutes_after_start = int(elapsed_seconds // 60)
    on_time_window = int(record.get("on_time_window_minutes", 0))
    grace_period = int(record["grace_period_minutes"])
    grace_used = min(
        max(0, minutes_after_start - on_time_window),
        max(0, grace_period - on_time_window),
    )
    late_minutes = math.ceil(
        max(0, elapsed_seconds - grace_period * 60) / 60
    )
    if check_in <= scheduled_start or check_in < scheduled_start + timedelta(
        minutes=on_time_window
    ):
        status = "ON_TIME"
    elif check_in <= scheduled_start + timedelta(minutes=grace_period):
        status = "GRACE_PERIOD"
    else:
        status = "LATE"
    early_departure_minutes = 0
    check_out_value = record.get("check_out")
    shift_end_value = record.get("shift_end") or record.get("end_time")
    if check_out_value and shift_end_value:
        check_out = datetime.fromisoformat(check_out_value)
        scheduled_end_day = date.fromisoformat(record["attendance_date"])
        shift_end = datetime_time.fromisoformat(shift_end_value)
        if shift_end <= shift_start:
            scheduled_end_day += timedelta(days=1)
        scheduled_end = datetime.combine(
            scheduled_end_day, shift_end, check_out.tzinfo
        )
        early_departure_minutes = math.ceil(
            max(0, (scheduled_end - check_out).total_seconds()) / 60
        )
    return {
        "attendance_status": status,
        "minutes_after_start": minutes_after_start,
        "grace_minutes_used": grace_used,
        "late_minutes": late_minutes,
        "late": int(status == "LATE"),
        "late_after_grace": int(status == "LATE"),
        "early_departure_minutes": early_departure_minutes,
    }


def _working_days(settings):
    try:
        days = json.loads(settings.get("working_days", DEFAULT_SETTINGS["working_days"]))
    except (TypeError, json.JSONDecodeError):
        raise ValueError("Configured working days are invalid.") from None
    if not isinstance(days, list) or any(day not in WEEKDAYS for day in days):
        raise ValueError("Configured working days are invalid.")
    return set(days)


def _holiday_dates(settings):
    try:
        holidays = json.loads(settings.get("holidays", "[]"))
        parsed = {date.fromisoformat(item) for item in holidays}
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("Configured holiday dates are invalid.") from None
    return parsed


@contextmanager
def connection(database_path=DATABASE_FILE):
    db = sqlite3.connect(database_path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize_database(database_path=DATABASE_FILE):
    os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
    with connection(database_path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                face_name TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                department TEXT NOT NULL DEFAULT '',
                designation TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                joining_date TEXT NOT NULL DEFAULT '',
                profile_photo TEXT NOT NULL DEFAULT '',
                shift_id INTEGER NOT NULL DEFAULT 1 REFERENCES shifts(shift_id),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'inactive')),
                consent_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shifts (
                shift_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                grace_period_minutes INTEGER NOT NULL DEFAULT 15
                    CHECK (grace_period_minutes >= 0),
                on_time_window_minutes INTEGER NOT NULL DEFAULT 0
                    CHECK (on_time_window_minutes >= 0),
                standard_work_hours REAL NOT NULL DEFAULT 8
                    CHECK (standard_work_hours > 0),
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                removed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY,
                employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                shift_id INTEGER NOT NULL REFERENCES shifts(shift_id),
                attendance_date TEXT NOT NULL,
                check_in TEXT NOT NULL,
                check_out TEXT,
                working_hours REAL,
                late INTEGER NOT NULL DEFAULT 0 CHECK (late IN (0, 1)),
                late_after_grace INTEGER NOT NULL DEFAULT 0
                    CHECK (late_after_grace IN (0, 1)),
                early_departure INTEGER NOT NULL DEFAULT 0
                    CHECK (early_departure IN (0, 1)),
                checkin_photo_path TEXT,
                checkout_photo_path TEXT,
                checkin_photo_timestamp TEXT,
                checkout_photo_timestamp TEXT,
                checkin_recognition_score REAL,
                checkout_recognition_score REAL,
                checkin_liveness_score REAL,
                checkout_liveness_score REAL,
                status TEXT NOT NULL DEFAULT 'present'
                    CHECK (status IN ('present', 'checked_out')),
                UNIQUE (employee_id, attendance_date, shift_id)
            );
            CREATE TABLE IF NOT EXISTS attendance_movements (
                movement_id INTEGER PRIMARY KEY,
                attendance_id INTEGER NOT NULL REFERENCES attendance(attendance_id)
                    ON DELETE CASCADE,
                action TEXT NOT NULL CHECK (action IN ('check_in', 'check_out')),
                occurred_at TEXT NOT NULL,
                photo_path TEXT,
                recognition_score REAL,
                liveness_score REAL
            );
            CREATE TABLE IF NOT EXISTS access_events (
                event_id INTEGER PRIMARY KEY,
                source_key TEXT UNIQUE,
                time TEXT NOT NULL,
                event TEXT NOT NULL,
                status TEXT NOT NULL,
                name TEXT NOT NULL,
                score TEXT NOT NULL DEFAULT '',
                method TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS leave_requests (
                leave_id INTEGER PRIMARY KEY,
                employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS employee_accounts (
                employee_id TEXT PRIMARY KEY REFERENCES employees(employee_id)
                    ON DELETE CASCADE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                annual_leave_days REAL CHECK (annual_leave_days >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
            CREATE TABLE IF NOT EXISTS hod_accounts (
                hod_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                department TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_hod_department
                ON hod_accounts(department);
            CREATE TABLE IF NOT EXISTS attendance_correction_requests (
                correction_id INTEGER PRIMARY KEY,
                employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                attendance_date TEXT NOT NULL,
                requested_check_in TEXT,
                requested_check_out TEXT,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewer_note TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_corrections_employee_date
                ON attendance_correction_requests(employee_id, attendance_date);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_active_unique
                ON attendance_correction_requests(employee_id, attendance_date)
                WHERE status IN ('pending', 'approved');
            CREATE TABLE IF NOT EXISTS attendance_correction_audit (
                audit_id INTEGER PRIMARY KEY,
                correction_id INTEGER NOT NULL REFERENCES attendance_correction_requests(correction_id)
                    ON DELETE CASCADE,
                action TEXT NOT NULL,
                actor_id TEXT NOT NULL DEFAULT '',
                actor_name TEXT NOT NULL DEFAULT '',
                actor_role TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_attendance_summary (
                summary_id INTEGER PRIMARY KEY,
                employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                attendance_date TEXT NOT NULL,
                first_check_in TEXT,
                final_checkout TEXT,
                working_hours REAL,
                break_duration REAL NOT NULL DEFAULT 0,
                offsite_duration REAL NOT NULL DEFAULT 0,
                attendance_status TEXT NOT NULL,
                punctuality_status TEXT NOT NULL DEFAULT 'NOT_CHECKED_IN',
                checkout_verification_status TEXT NOT NULL DEFAULT 'pending',
                correction_approval_status TEXT NOT NULL DEFAULT 'none',
                early_departure_minutes INTEGER NOT NULL DEFAULT 0,
                late_minutes INTEGER NOT NULL DEFAULT 0,
                last_reconciled_at TEXT NOT NULL,
                UNIQUE (employee_id, attendance_date)
            );
            CREATE INDEX IF NOT EXISTS idx_daily_summary_date_department
                ON daily_attendance_summary(attendance_date, employee_id);
            CREATE TABLE IF NOT EXISTS offsite_meeting_records (
                meeting_id INTEGER PRIMARY KEY,
                employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                attendance_date TEXT NOT NULL,
                departure_time TEXT NOT NULL,
                return_time TEXT,
                meeting_title TEXT NOT NULL,
                meeting_details TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
                duration_hours REAL,
                submitted_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewer_id TEXT NOT NULL DEFAULT '',
                reviewer_name TEXT NOT NULL DEFAULT '',
                reviewer_role TEXT NOT NULL DEFAULT '',
                reviewer_note TEXT NOT NULL DEFAULT '',
                UNIQUE (employee_id, attendance_date, departure_time)
            );
            CREATE TABLE IF NOT EXISTS app_notifications (
                notification_id INTEGER PRIMARY KEY,
                recipient_role TEXT NOT NULL,
                recipient_id TEXT NOT NULL DEFAULT '',
                employee_id TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0 CHECK (is_read IN (0, 1)),
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notifications_recipient
                ON app_notifications(recipient_role, recipient_id, is_read);
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chatbot_messages (
                message_id INTEGER PRIMARY KEY,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                photos_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chatbot_messages_session
                ON chatbot_messages(session_key, message_id);
            CREATE TABLE IF NOT EXISTS chatbot_context (
                session_key TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_attendance_date
                ON attendance(attendance_date);
            CREATE INDEX IF NOT EXISTS idx_attendance_employee_date
                ON attendance(employee_id, attendance_date);
            CREATE INDEX IF NOT EXISTS idx_access_time
                ON access_events(time);
            """
        )
        employee_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(employees)")
        }
        if "shift_id" not in employee_columns:
            db.execute(
                "ALTER TABLE employees ADD COLUMN shift_id INTEGER NOT NULL DEFAULT 1"
            )
        for key, value in DEFAULT_SETTINGS.items():
            db.execute(
                "INSERT OR IGNORE INTO app_settings(setting_key, setting_value) VALUES (?, ?)",
                (key, value),
            )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        grace_migration = db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (2, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        if grace_migration.rowcount:
            db.execute(
                "UPDATE shifts SET grace_period_minutes = 15 "
                "WHERE shift_id = 1 AND grace_period_minutes = 10"
            )
            db.execute(
                """UPDATE app_settings SET setting_value = '15'
                   WHERE setting_key = 'grace_period_minutes' AND setting_value = '10'"""
            )
        default_shift_migration = db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        if default_shift_migration.rowcount:
            db.execute(
                """INSERT OR IGNORE INTO shifts
                   (shift_id, name, start_time, end_time, grace_period_minutes,
                    standard_work_hours, active)
                   VALUES (1, 'Default', '09:00', '18:00', 15, 8, 1)"""
            )
        shift_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(shifts)")
        }
        if "removed_at" not in shift_columns:
            db.execute("ALTER TABLE shifts ADD COLUMN removed_at TEXT")
        if "on_time_window_minutes" not in shift_columns:
            db.execute(
                """ALTER TABLE shifts ADD COLUMN on_time_window_minutes INTEGER
                   NOT NULL DEFAULT 0 CHECK (on_time_window_minutes >= 0)"""
            )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (4, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        attendance_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(attendance)")
        }
        if "late_after_grace" not in attendance_columns:
            db.execute(
                """ALTER TABLE attendance ADD COLUMN late_after_grace INTEGER NOT NULL
                   DEFAULT 0 CHECK (late_after_grace IN (0, 1))"""
            )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (5, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        photo_columns = {
            "checkin_photo_path": "TEXT",
            "checkout_photo_path": "TEXT",
            "checkin_photo_timestamp": "TEXT",
            "checkout_photo_timestamp": "TEXT",
            "checkin_recognition_score": "REAL",
            "checkout_recognition_score": "REAL",
            "checkin_liveness_score": "REAL",
            "checkout_liveness_score": "REAL",
        }
        for column, sql_type in photo_columns.items():
            if column not in attendance_columns:
                db.execute(f"ALTER TABLE attendance ADD COLUMN {column} {sql_type}")
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (6, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (7, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        movement_columns = {
            row["name"] for row in db.execute(
                "PRAGMA table_info(attendance_movements)"
            )
        }
        if "liveness_score" not in movement_columns:
            db.execute(
                "ALTER TABLE attendance_movements ADD COLUMN liveness_score REAL"
            )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (8, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (9, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        attendance_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(attendance)")
        }
        if "missing_checkout_pending" not in attendance_columns:
            db.execute(
                "ALTER TABLE attendance ADD COLUMN missing_checkout_pending INTEGER NOT NULL DEFAULT 0"
                " CHECK (missing_checkout_pending IN (0, 1))"
            )
        correction_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(attendance_correction_requests)")
        }
        correction_additions = {
            "request_type": "TEXT NOT NULL DEFAULT 'Forgot to checkout'",
            "supporting_document": "TEXT NOT NULL DEFAULT ''",
            "original_check_in": "TEXT",
            "original_check_out": "TEXT",
            "approved_check_out": "TEXT",
            "approver_id": "TEXT NOT NULL DEFAULT ''",
            "approver_name": "TEXT NOT NULL DEFAULT ''",
            "approver_role": "TEXT NOT NULL DEFAULT ''",
        }
        for column, sql_type in correction_additions.items():
            if column not in correction_columns:
                db.execute(
                    f"ALTER TABLE attendance_correction_requests ADD COLUMN {column} {sql_type}"
                )
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_active_unique "
            "ON attendance_correction_requests(employee_id, attendance_date) "
            "WHERE status IN ('pending', 'approved')"
        )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (10, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (11, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
        chatbot_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(chatbot_messages)")
        }
        if "photos_json" not in chatbot_columns:
            db.execute(
                "ALTER TABLE chatbot_messages ADD COLUMN photos_json TEXT NOT NULL DEFAULT '[]'"
            )
        db.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (12, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
    reconcile_daily_attendance_summaries(database_path)
    if os.name == "posix":
        os.chmod(database_path, 0o600)


def _next_employee_id(db):
    rows = db.execute(
        "SELECT employee_id FROM employees WHERE employee_id GLOB 'EMP-[0-9]*'"
    ).fetchall()
    highest = max(
        (int(row["employee_id"][4:]) for row in rows
         if row["employee_id"][4:].isdigit()),
        default=0,
    )
    return f"EMP-{highest + 1:06d}"


def sync_legacy_staff(staff, staff_meta, database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        default_shift = db.execute(
            """SELECT shift_id FROM shifts
               WHERE active = 1 AND removed_at IS NULL
               ORDER BY shift_id LIMIT 1"""
        ).fetchone()
        if staff and default_shift is None:
            raise ValueError("Create or enable an active shift before adding employees.")
        for name in sorted(staff):
            meta = staff_meta.get(name, {})
            status = "active" if meta.get("active", True) else "inactive"
            db.execute(
                """INSERT OR IGNORE INTO employees
                   (employee_id, face_name, full_name, shift_id, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_next_employee_id(db), name, name, default_shift["shift_id"], status, now, now),
            )
            db.execute(
                """UPDATE employees SET status = ?, updated_at = ?
                   WHERE face_name = ?""",
                (status, now, name),
            )


def import_legacy_entries(csv_path, database_path=DATABASE_FILE):
    if not os.path.isfile(csv_path):
        return 0
    imported = 0
    occurrences = {}
    unkeyed_counts = {}
    imported_counts = {}
    with connection(database_path) as db:
        with open(csv_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                normalized = {
                    "time": row.get("time", "") or "",
                    "event": row.get("event", "") or "entry",
                    "status": row.get("status", "") or "granted",
                    "name": row.get("name", "") or "",
                    "score": row.get("score", "") or "",
                    "method": row.get("method", "") or "face",
                    "details": row.get("details", "") or "",
                }
                content = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
                occurrence_key = hashlib.sha256(content.encode("utf-8")).hexdigest()
                occurrences[occurrence_key] = occurrences.get(occurrence_key, 0) + 1
                if occurrence_key not in unkeyed_counts:
                    unkeyed_counts[occurrence_key] = db.execute(
                        """SELECT COUNT(*) FROM access_events
                           WHERE source_key IS NULL AND time = ? AND event = ?
                             AND status = ? AND name = ? AND score = ?
                             AND method = ? AND details = ?""",
                        tuple(normalized.values()),
                    ).fetchone()[0]
                imported_counts[occurrence_key] = imported_counts.get(occurrence_key, 0) + 1
                if imported_counts[occurrence_key] <= unkeyed_counts[occurrence_key]:
                    continue
                source_key = f"legacy:{occurrence_key}:{occurrences[occurrence_key]}"
                cursor = db.execute(
                    """INSERT OR IGNORE INTO access_events
                       (source_key, time, event, status, name, score, method, details)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source_key, *normalized.values()),
                )
                imported += cursor.rowcount
    return imported


def log_access_event(entry, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.execute(
            """INSERT INTO access_events
               (time, event, status, name, score, method, details)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            tuple(entry[key] for key in
                  ("time", "event", "status", "name", "score", "method", "details")),
        )


def access_event_list(limit=10000, database_path=DATABASE_FILE):
    limit = max(1, min(int(limit), 100000))
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT time, event, status, name, score, method, details
               FROM access_events ORDER BY event_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def settings(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        values = dict(DEFAULT_SETTINGS)
        values.update({
            row["setting_key"]: row["setting_value"]
            for row in db.execute("SELECT setting_key, setting_value FROM app_settings")
        })
        return values


def update_settings(values, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.executemany(
            """INSERT INTO app_settings(setting_key, setting_value) VALUES (?, ?)
               ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value""",
            [(key, str(value)) for key, value in values.items()],
        )


def save_recognition_proof(token, proof, database_path=DATABASE_FILE):
    key = "recognition_proof_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    value = json.dumps(proof, separators=(",", ":"))
    with connection(database_path) as db:
        rows = db.execute(
            "SELECT setting_key, setting_value FROM app_settings WHERE setting_key LIKE 'recognition_proof_%'"
        ).fetchall()
        expired = []
        now = time.time()
        for row in rows:
            try:
                stored = json.loads(row["setting_value"])
                if float(stored.get("expires", 0)) < now:
                    expired.append(row["setting_key"])
            except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
                expired.append(row["setting_key"])
        for expired_key in expired:
            db.execute("DELETE FROM app_settings WHERE setting_key = ?", (expired_key,))
        db.execute(
            "INSERT INTO app_settings(setting_key, setting_value) VALUES (?, ?) "
            "ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value",
            (key, value),
        )


def consume_recognition_proofs(tokens, database_path=DATABASE_FILE):
    results = {}
    with connection(database_path) as db:
        for token in tokens:
            key = "recognition_proof_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
            row = db.execute(
                "DELETE FROM app_settings WHERE setting_key = ? RETURNING setting_value",
                (key,),
            ).fetchone()
            if row:
                try:
                    proof = json.loads(row["setting_value"])
                    if float(proof.get("expires", 0)) >= time.time():
                        results[token] = (
                            proof["name"], float(proof["score"]),
                            float(proof["expires"]), float(proof["recognized_at"]),
                        )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError, AttributeError):
                    continue
    return results


def employees(active_only=False, database_path=DATABASE_FILE):
    query = "SELECT * FROM employees"
    if active_only:
        query += " WHERE status = 'active'"
    query += " ORDER BY full_name COLLATE NOCASE"
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(query)]


def employee_by_face_name(face_name, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            "SELECT * FROM employees WHERE face_name = ?", (face_name,)
        ).fetchone()
        return dict(row) if row else None


def employee_by_id(employee_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
        return dict(row) if row else None


def employee_account(employee_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            """SELECT employee_id, password_salt, password_hash, is_active,
                      annual_leave_days, created_at, updated_at, last_login_at
               FROM employee_accounts WHERE employee_id = ?""",
            (employee_id,),
        ).fetchone()
        return dict(row) if row else None


def employee_account_for_login(identifier, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT a.employee_id, a.password_salt, a.password_hash, a.is_active,
                      e.status AS employee_status, e.email
               FROM employee_accounts a
               JOIN employees e ON e.employee_id = a.employee_id
               WHERE e.employee_id = ? COLLATE NOCASE
                  OR (trim(e.email) != ''
                      AND lower(trim(e.email)) = lower(trim(?)))""",
            (identifier, identifier),
        ).fetchall()
        return dict(rows[0]) if len(rows) == 1 else None


def employee_account_list(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [
            dict(row) for row in db.execute(
                """SELECT e.employee_id, e.full_name, e.department, e.designation,
                          e.email, e.status AS employee_status,
                          a.is_active AS account_active, a.annual_leave_days,
                          a.created_at AS account_created_at,
                          a.last_login_at
                   FROM employees e
                   LEFT JOIN employee_accounts a ON a.employee_id = e.employee_id
                   ORDER BY e.full_name COLLATE NOCASE"""
            )
        ]


def create_employee_account(
    employee_id, password_salt, password_hash, annual_leave_days=None,
    database_path=DATABASE_FILE,
):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        cursor = db.execute(
            """INSERT INTO employee_accounts
               (employee_id, password_salt, password_hash, annual_leave_days,
                created_at, updated_at)
               SELECT employee_id, ?, ?, ?, ?, ? FROM employees
               WHERE employee_id = ?""",
            (
                password_salt, password_hash, annual_leave_days, now, now,
                employee_id,
            ),
        )
        return cursor.rowcount > 0


def update_employee_account(employee_id, values, database_path=DATABASE_FILE):
    allowed = {"password_salt", "password_hash", "is_active", "annual_leave_days"}
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        return False
    updates["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with connection(database_path) as db:
        cursor = db.execute(
            f"UPDATE employee_accounts SET {assignments} WHERE employee_id = ?",
            (*updates.values(), employee_id),
        )
        return cursor.rowcount > 0


def update_employee_last_login(employee_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return db.execute(
            """UPDATE employee_accounts SET last_login_at = ?
               WHERE employee_id = ?""",
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                employee_id,
            ),
        ).rowcount > 0


def _next_hod_id(db):
    rows = db.execute("SELECT hod_id FROM hod_accounts WHERE hod_id GLOB 'HOD-[0-9]*'").fetchall()
    highest = max(
        (int(row["hod_id"][4:]) for row in rows if row["hod_id"][4:].isdigit()),
        default=0,
    )
    return f"HOD-{highest + 1:06d}"


def hod_account_list(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(
            """SELECT hod_id, full_name, email, department, is_active,
                      created_at, updated_at, last_login_at
               FROM hod_accounts
               ORDER BY full_name COLLATE NOCASE"""
        )]


def hod_account_for_login(identifier, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT * FROM hod_accounts
               WHERE hod_id = ? COLLATE NOCASE
                  OR lower(trim(email)) = lower(trim(?))""",
            (identifier, identifier),
        ).fetchall()
        return dict(rows[0]) if len(rows) == 1 else None


def hod_account_by_id(hod_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute("SELECT * FROM hod_accounts WHERE hod_id = ?", (hod_id,)).fetchone()
        return dict(row) if row else None


def create_hod_account(full_name, email, department, password_salt, password_hash,
                       database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        hod_id = _next_hod_id(db)
        db.execute(
            """INSERT INTO hod_accounts
               (hod_id, full_name, email, department, password_salt, password_hash,
                is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (hod_id, full_name, email, department, password_salt, password_hash, now, now),
        )
        return hod_id


def update_hod_account(hod_id, values, database_path=DATABASE_FILE):
    allowed = {"full_name", "email", "department", "password_salt", "password_hash", "is_active"}
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        return False
    updates["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with connection(database_path) as db:
        cursor = db.execute(
            f"UPDATE hod_accounts SET {assignments} WHERE hod_id = ?",
            (*updates.values(), hod_id),
        )
        return cursor.rowcount > 0


def delete_hod_account(hod_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return db.execute("DELETE FROM hod_accounts WHERE hod_id = ?", (hod_id,)).rowcount > 0


def update_hod_last_login(hod_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return db.execute(
            "UPDATE hod_accounts SET last_login_at = ? WHERE hod_id = ?",
            (datetime.now().astimezone().isoformat(timespec="seconds"), hod_id),
        ).rowcount > 0


def update_employee(employee_id, values, database_path=DATABASE_FILE):
    allowed = {
        "full_name", "department", "designation", "email", "phone",
        "joining_date", "status", "consent_at", "shift_id",
    }
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        return False
    updates["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with connection(database_path) as db:
        cursor = db.execute(
            f"UPDATE employees SET {assignments} WHERE employee_id = ?",
            (*updates.values(), employee_id),
        )
        return cursor.rowcount > 0


def upsert_enrolled_employee(face_name, values, database_path=DATABASE_FILE):
    now = datetime.now(ZoneInfo("Asia/Qatar")).isoformat(timespec="seconds")
    with connection(database_path) as db:
        default_shift = db.execute(
            """SELECT shift_id FROM shifts WHERE active = 1
               AND removed_at IS NULL
               ORDER BY shift_id LIMIT 1"""
        ).fetchone()
    if default_shift is None:
        raise ValueError("Create or enable an active shift before enrolling employees.")
    fields = {
        "full_name": values.get("full_name") or face_name,
        "department": values.get("department", ""),
        "designation": values.get("designation", ""),
        "email": values.get("email", ""),
        "phone": values.get("phone", ""),
        "joining_date": values.get("joining_date", ""),
        "profile_photo": values.get("profile_photo", ""),
        "shift_id": values.get("shift_id", default_shift["shift_id"]),
        "consent_at": now,
        "status": "active",
        "updated_at": now,
    }
    with connection(database_path) as db:
        current = db.execute(
            "SELECT employee_id FROM employees WHERE face_name = ?", (face_name,)
        ).fetchone()
        if current:
            updates = {
                "profile_photo": fields["profile_photo"],
                "consent_at": fields["consent_at"],
                "updated_at": fields["updated_at"],
            }
            for key in ("department", "designation", "email", "phone", "joining_date"):
                if fields[key]:
                    updates[key] = fields[key]
            assignments = ", ".join(f"{key} = ?" for key in updates)
            db.execute(
                f"UPDATE employees SET {assignments} WHERE employee_id = ?",
                (*updates.values(), current["employee_id"]),
            )
            return current["employee_id"]
        employee_id = _next_employee_id(db)
        db.execute(
            """INSERT INTO employees
               (employee_id, face_name, full_name, department, designation, email,
                phone, joining_date, profile_photo, shift_id, status, consent_at,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                employee_id, face_name, fields["full_name"], fields["department"],
                fields["designation"], fields["email"], fields["phone"],
                fields["joining_date"], fields["profile_photo"], fields["shift_id"],
                fields["status"], fields["consent_at"], now, now,
            ),
        )
        return employee_id


def set_employee_face_name(old_name, new_name, database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        cursor = db.execute(
            "UPDATE employees SET face_name = ?, full_name = ?, updated_at = ? WHERE face_name = ?",
            (new_name, new_name, now, old_name),
        )
        return cursor.rowcount > 0


def deactivate_employee(face_name, database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        db.execute(
            "UPDATE employees SET status = 'inactive', updated_at = ? WHERE face_name = ?",
            (now, face_name),
        )


def erase_employee_biometrics(face_name, database_path=DATABASE_FILE):
    now = datetime.now(ZoneInfo("Asia/Qatar")).isoformat(timespec="seconds")
    with connection(database_path) as db:
        db.execute(
            """UPDATE employees SET status = 'inactive', consent_at = NULL,
               profile_photo = '', updated_at = ? WHERE face_name = ?""",
            (now, face_name),
        )


def list_shifts(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(
            """SELECT s.*,
                      (SELECT COUNT(*) FROM employees e WHERE e.shift_id = s.shift_id)
                          AS employee_count,
                      (SELECT COUNT(*) FROM attendance a WHERE a.shift_id = s.shift_id)
                          AS attendance_count
               FROM shifts s WHERE s.removed_at IS NULL ORDER BY s.shift_id"""
        )]


def create_shift(name, start_time, end_time, grace_period_minutes, standard_work_hours,
                 database_path=DATABASE_FILE, on_time_window_minutes=0):
    if on_time_window_minutes > grace_period_minutes:
        raise ValueError("On-time window cannot exceed the late-arrival cutoff.")
    with connection(database_path) as db:
        cursor = db.execute(
            """INSERT INTO shifts
               (name, start_time, end_time, grace_period_minutes,
                on_time_window_minutes, standard_work_hours)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                name, start_time, end_time, grace_period_minutes,
                on_time_window_minutes, standard_work_hours,
            ),
        )
        return cursor.lastrowid


def update_shift_times(shift_id, start_time, end_time, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        cursor = db.execute(
            """UPDATE shifts SET start_time = ?, end_time = ?
               WHERE shift_id = ? AND removed_at IS NULL""",
            (start_time, end_time, shift_id),
        )
        return cursor.rowcount > 0


def update_shift_configuration(
    shift_id, start_time, end_time, grace_period_minutes, database_path=DATABASE_FILE,
    on_time_window_minutes=0,
):
    if on_time_window_minutes > grace_period_minutes:
        raise ValueError("On-time window cannot exceed the late-arrival cutoff.")
    with connection(database_path) as db:
        cursor = db.execute(
            """UPDATE shifts SET start_time = ?, end_time = ?,
               grace_period_minutes = ?, on_time_window_minutes = ?
               WHERE shift_id = ? AND removed_at IS NULL""",
            (
                start_time, end_time, grace_period_minutes,
                on_time_window_minutes, shift_id,
            ),
        )
        return cursor.rowcount > 0


def remove_shift(shift_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        shift = db.execute(
            """SELECT shift_id, active FROM shifts
               WHERE shift_id = ? AND removed_at IS NULL""",
            (shift_id,),
        ).fetchone()
        if shift is None:
            return None
        employee_count = db.execute(
            "SELECT COUNT(*) FROM employees WHERE shift_id = ?", (shift_id,)
        ).fetchone()[0]
        if employee_count:
            raise ValueError(
                "Assign every employee to their replacement shift before removing this shift."
            )
        replacement = db.execute(
            """SELECT shift_id FROM shifts
               WHERE active = 1 AND removed_at IS NULL AND shift_id != ?
               ORDER BY shift_id LIMIT 1""",
            (shift_id,),
        ).fetchone()
        if shift["active"] and replacement is None:
            raise ValueError("Add or enable a replacement shift first.")
        attendance_count = db.execute(
            "SELECT COUNT(*) FROM attendance WHERE shift_id = ?", (shift_id,)
        ).fetchone()[0]
        if attendance_count:
            db.execute(
                """UPDATE shifts SET active = 0, removed_at = ?
                   WHERE shift_id = ?""",
                (datetime.now().astimezone().isoformat(timespec="seconds"), shift_id),
            )
        else:
            db.execute("DELETE FROM shifts WHERE shift_id = ?", (shift_id,))
        return {
            "attendance_preserved": bool(attendance_count),
        }


def set_shift_active(shift_id, active, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        shift = db.execute(
            """SELECT active FROM shifts
               WHERE shift_id = ? AND removed_at IS NULL""",
            (shift_id,),
        ).fetchone()
        if shift is None:
            return False
        if not active and shift["active"] and db.execute(
            "SELECT COUNT(*) FROM shifts WHERE active = 1 AND removed_at IS NULL"
        ).fetchone()[0] == 1:
            raise ValueError("At least one active shift must remain.")
        db.execute(
            "UPDATE shifts SET active = ? WHERE shift_id = ?",
            (int(active), shift_id),
        )
        return True


def enable_shift(shift_id, database_path=DATABASE_FILE):
    return set_shift_active(shift_id, True, database_path)


def set_employee_shift(employee_id, shift_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        cursor = db.execute(
            "UPDATE employees SET shift_id = ? WHERE employee_id = ?",
            (shift_id, employee_id),
        )
        return cursor.rowcount > 0


def create_leave_request(employee_id, start_date, end_date, reason,
                         database_path=DATABASE_FILE):
    with connection(database_path) as db:
        cursor = db.execute(
            """INSERT INTO leave_requests
               (employee_id, start_date, end_date, reason, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                employee_id, start_date, end_date, reason,
                datetime.now(ZoneInfo("Asia/Qatar")).isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def leave_requests(database_path=DATABASE_FILE, employee_id=""):
    employee_filter = "WHERE l.employee_id = ?" if employee_id else ""
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(
            """SELECT l.*, e.full_name, e.department FROM leave_requests l
               JOIN employees e ON e.employee_id = l.employee_id """
            + employee_filter + " ORDER BY l.created_at DESC LIMIT 1000",
            (employee_id,) if employee_id else (),
        )]


def leave_balance(employee_id, year, database_path=DATABASE_FILE):
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    account = employee_account(employee_id, database_path)
    if account is None:
        return None
    requests = leave_requests(database_path, employee_id)
    used_days = 0
    for request in requests:
        if request["status"] != "approved":
            continue
        start = max(date.fromisoformat(request["start_date"]), year_start)
        end = min(date.fromisoformat(request["end_date"]), year_end)
        if end >= start:
            used_days += (end - start).days + 1
    entitled_days = account["annual_leave_days"]
    return {
        "year": year,
        "entitled_days": entitled_days,
        "used_days": used_days,
        "remaining_days": (
            max(0, entitled_days - used_days)
            if entitled_days is not None else None
        ),
    }


def update_leave_status(leave_id, status, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        cursor = db.execute(
            """UPDATE leave_requests SET status = ?, reviewed_at = ?
               WHERE leave_id = ? AND status = 'pending'""",
            (
                status, datetime.now(ZoneInfo("Asia/Qatar")).isoformat(timespec="seconds"),
                leave_id,
            ),
        )
        return cursor.rowcount > 0


def create_attendance_correction_request(
    employee_id, attendance_date, requested_check_in, requested_check_out, reason,
    request_type="Forgot to checkout", supporting_document="", database_path=DATABASE_FILE,
):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        existing = db.execute(
            """SELECT correction_id FROM attendance_correction_requests
               WHERE employee_id = ? AND attendance_date = ?
                 AND status IN ('pending', 'approved') LIMIT 1""",
            (employee_id, attendance_date),
        ).fetchone()
        if existing:
            return None
        attendance = db.execute(
            """SELECT check_in, check_out FROM attendance
               WHERE employee_id = ? AND attendance_date = ?
               ORDER BY attendance_id LIMIT 1""",
            (employee_id, attendance_date),
        ).fetchone()
        cursor = db.execute(
            """INSERT INTO attendance_correction_requests
               (employee_id, attendance_date, requested_check_in,
                requested_check_out, reason, request_type, supporting_document,
                original_check_in, original_check_out, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                employee_id, attendance_date, requested_check_in,
                requested_check_out, reason, request_type, supporting_document,
                attendance["check_in"] if attendance else None,
                attendance["check_out"] if attendance else None,
                now,
            ),
        )
        correction_id = cursor.lastrowid
        db.execute(
            """INSERT INTO attendance_correction_audit
               (correction_id, action, actor_id, actor_name, actor_role, details, created_at)
               VALUES (?, 'submitted', ?, '', 'employee', ?, ?)""",
            (correction_id, employee_id, json.dumps({
                "attendance_date": attendance_date,
                "requested_check_out": requested_check_out,
                "request_type": request_type,
            }), now),
        )
        return correction_id


def attendance_correction_requests(database_path=DATABASE_FILE, employee_id="", department="", status=""):
    clauses = []
    params = []
    if employee_id:
        clauses.append("r.employee_id = ?")
        params.append(employee_id)
    if department:
        clauses.append("lower(trim(e.department)) = lower(trim(?))")
        params.append(department)
    if status:
        clauses.append("r.status = ?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connection(database_path) as db:
        return [
            dict(row) for row in db.execute(
                """SELECT r.*, e.full_name, e.department
                   FROM attendance_correction_requests r
                   JOIN employees e ON e.employee_id = r.employee_id"""
                + where + " ORDER BY r.created_at DESC LIMIT 1000",
                tuple(params),
            )
        ]


def update_attendance_correction_status(
    correction_id, status, reviewer_note="", database_path=DATABASE_FILE
):
    with connection(database_path) as db:
        cursor = db.execute(
            """UPDATE attendance_correction_requests
               SET status = ?, reviewer_note = ?, reviewed_at = ?
               WHERE correction_id = ? AND status = 'pending'""",
            (
                status, reviewer_note,
                datetime.now().astimezone().isoformat(timespec="seconds"),
                correction_id,
            ),
        )
        return cursor.rowcount > 0


def review_attendance_correction(
    correction_id, status, reviewer_note="", approver_id="", approver_name="",
    approver_role="", database_path=DATABASE_FILE
):
    timezone = ZoneInfo(settings(database_path=database_path)["timezone"])
    with connection(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        correction = db.execute(
            """SELECT * FROM attendance_correction_requests
               WHERE correction_id = ? AND status = 'pending'""",
            (correction_id,),
        ).fetchone()
        if correction is None:
            return False
        check_out = None
        attendance = None
        if status == "approved":
            employee = db.execute(
                """SELECT employee_id, shift_id FROM employees
                   WHERE employee_id = ?""",
                (correction["employee_id"],),
            ).fetchone()
            attendance = db.execute(
                """SELECT * FROM attendance
                   WHERE employee_id = ? AND attendance_date = ?
                   ORDER BY attendance_id LIMIT 1""",
                (correction["employee_id"], correction["attendance_date"]),
            ).fetchone()
            shift_id = attendance["shift_id"] if attendance else employee["shift_id"]
            shift = db.execute(
                """SELECT start_time, end_time, grace_period_minutes,
                          on_time_window_minutes
                   FROM shifts WHERE shift_id = ?""",
                (shift_id,),
            ).fetchone()
            if shift is None:
                raise ValueError("The employee has no usable shift for that date.")
            if attendance is None and not correction["requested_check_in"]:
                raise ValueError(
                    "A check-in time is required to create a missing attendance record."
                )
            attendance_day = date.fromisoformat(correction["attendance_date"])
            check_in = (
                datetime.combine(
                    attendance_day,
                    datetime_time.fromisoformat(correction["requested_check_in"]),
                    timezone,
                )
                if correction["requested_check_in"]
                else datetime.fromisoformat(attendance["check_in"])
            )
            if correction["requested_check_out"]:
                check_out_time = datetime_time.fromisoformat(
                    correction["requested_check_out"]
                )
                check_out_day = attendance_day
                if check_out_time <= check_in.timetz().replace(tzinfo=None):
                    check_out_day += timedelta(days=1)
                check_out = datetime.combine(check_out_day, check_out_time, timezone)
            else:
                check_out = (
                    datetime.fromisoformat(attendance["check_out"])
                    if attendance and attendance["check_out"] else None
                )
            if check_out is not None and check_out <= check_in:
                raise ValueError("Corrected check-out must be later than check-in.")
            working_hours = (
                round((check_out - check_in).total_seconds() / 3600, 2)
                if check_out else None
            )
            classification = classify_attendance({
                "attendance_date": correction["attendance_date"],
                "check_in": check_in.isoformat(timespec="seconds"),
                "check_out": check_out.isoformat(timespec="seconds") if check_out else None,
                "shift_start": shift["start_time"],
                "shift_end": shift["end_time"],
                "grace_period_minutes": shift["grace_period_minutes"],
                "on_time_window_minutes": shift["on_time_window_minutes"],
            })
            attendance_values = (
                check_in.isoformat(timespec="seconds"),
                check_out.isoformat(timespec="seconds") if check_out else None,
                working_hours,
                classification["late"],
                classification["late_after_grace"],
                int(classification["early_departure_minutes"] > 0),
                "checked_out" if check_out else "present",
            )
            if attendance:
                db.execute(
                    """UPDATE attendance
                       SET check_in = ?, check_out = ?, working_hours = ?,
                           late = ?, late_after_grace = ?, early_departure = ?,
                           status = ?
                       WHERE attendance_id = ?""",
                    (*attendance_values, attendance["attendance_id"]),
                )
            else:
                db.execute(
                    """INSERT INTO attendance
                       (employee_id, shift_id, attendance_date, check_in, check_out,
                        working_hours, late, late_after_grace, early_departure, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        correction["employee_id"], shift_id,
                        correction["attendance_date"], *attendance_values,
                    ),
                )
        reviewed_at = datetime.now(timezone).isoformat(timespec="seconds")
        approved_check_out = (
            check_out.isoformat(timespec="seconds")
            if status == "approved" and 'check_out' in locals() and check_out else None
        )
        cursor = db.execute(
            """UPDATE attendance_correction_requests
               SET status = ?, reviewer_note = ?, reviewed_at = ?,
                   approved_check_out = ?, approver_id = ?, approver_name = ?, approver_role = ?,
                   original_check_in = COALESCE(original_check_in, ?),
                   original_check_out = COALESCE(original_check_out, ?)
               WHERE correction_id = ? AND status = 'pending'""",
            (
                status, reviewer_note, reviewed_at, approved_check_out,
                approver_id, approver_name, approver_role,
                correction["original_check_in"] or (attendance["check_in"] if attendance else None),
                correction["original_check_out"] or (attendance["check_out"] if attendance else None),
                correction_id,
            ),
        )
        if cursor.rowcount > 0:
            if status == "approved":
                db.execute(
                    "UPDATE attendance SET missing_checkout_pending = 0 WHERE employee_id = ? AND attendance_date = ?",
                    (correction["employee_id"], correction["attendance_date"]),
                )
            db.execute(
                """INSERT INTO attendance_correction_audit
                   (correction_id, action, actor_id, actor_name, actor_role, details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (correction_id, status, approver_id, approver_name, approver_role,
                 json.dumps({
                     "reviewer_note": reviewer_note,
                     "approved_check_out": approved_check_out,
                 }), reviewed_at),
            )
            db.execute(
                """INSERT INTO app_notifications
                   (recipient_role, recipient_id, employee_id, category, title, message, created_at)
                   VALUES ('employee', ?, ?, 'attendance_correction', ?, ?, ?)""",
                (
                    correction["employee_id"], correction["employee_id"],
                    f"Attendance correction {status}",
                    f"Your correction for {correction['attendance_date']} was {status}.",
                    reviewed_at,
                ),
            )
        return cursor.rowcount > 0


def create_offsite_meeting(employee_id, attendance_date, departure_time,
                           meeting_title, meeting_details="", location="",
                           return_time="", database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        cursor = db.execute(
            """INSERT INTO offsite_meeting_records
               (employee_id, attendance_date, departure_time, return_time,
                meeting_title, meeting_details, location, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (employee_id, attendance_date, departure_time, return_time or None,
             meeting_title, meeting_details, location, now),
        )
        meeting_id = cursor.lastrowid
        return meeting_id


def offsite_meeting_records(database_path=DATABASE_FILE, employee_id="", department="", status=""):
    clauses = []
    params = []
    if employee_id:
        clauses.append("m.employee_id = ?")
        params.append(employee_id)
    if department:
        clauses.append("lower(trim(e.department)) = lower(trim(?))")
        params.append(department)
    if status:
        clauses.append("m.status = ?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT m.*, e.full_name, e.department
               FROM offsite_meeting_records m
               JOIN employees e ON e.employee_id = m.employee_id"""
            + where + " ORDER BY m.attendance_date DESC, m.departure_time DESC",
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]


def review_offsite_meeting(meeting_id, status, reviewer_note="", reviewer_id="",
                           reviewer_name="", reviewer_role="", database_path=DATABASE_FILE):
    now = datetime.now(ZoneInfo(settings(database_path=database_path)["timezone"])).isoformat(timespec="seconds")
    with connection(database_path) as db:
        meeting = db.execute(
            "SELECT * FROM offsite_meeting_records WHERE meeting_id = ? AND status = 'pending'",
            (meeting_id,),
        ).fetchone()
        if meeting is None:
            return False
        duration = None
        if meeting["return_time"]:
            duration = round(
                max(0, (
                    datetime.fromisoformat(meeting["return_time"])
                    - datetime.fromisoformat(meeting["departure_time"])
                ).total_seconds()) / 3600,
                2,
            )
        cursor = db.execute(
            """UPDATE offsite_meeting_records
               SET status=?, duration_hours=?, reviewed_at=?, reviewer_id=?,
                   reviewer_name=?, reviewer_role=?, reviewer_note=?
               WHERE meeting_id=? AND status='pending'""",
            (status, duration, now, reviewer_id, reviewer_name, reviewer_role,
             reviewer_note, meeting_id),
        )
        if cursor.rowcount:
            db.execute(
                """INSERT INTO app_notifications
                   (recipient_role, recipient_id, employee_id, category, title, message, created_at)
                   VALUES ('employee', ?, ?, 'offsite_meeting', ?, ?, ?)""",
                (meeting["employee_id"], meeting["employee_id"],
                 f"Off-site meeting request {status}",
                 f"Your meeting for {meeting['attendance_date']} was {status}.", now),
            )
        return cursor.rowcount > 0


def add_notification(recipient_role, recipient_id, employee_id, category, title, message,
                     database_path=DATABASE_FILE):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        existing = db.execute(
            """SELECT notification_id FROM app_notifications
               WHERE recipient_role = ? AND recipient_id = ? AND employee_id = ?
                 AND category = ? AND title = ? AND message = ? LIMIT 1""",
            (recipient_role, recipient_id, employee_id, category, title, message),
        ).fetchone()
        if existing:
            return existing["notification_id"]
        cursor = db.execute(
            """INSERT INTO app_notifications
               (recipient_role, recipient_id, employee_id, category, title, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (recipient_role, recipient_id, employee_id, category, title, message, now),
        )
        return cursor.lastrowid


def notifications(recipient_role, recipient_id="", database_path=DATABASE_FILE):
    clauses = ["recipient_role = ?"]
    params = [recipient_role]
    if recipient_id:
        clauses.append("recipient_id = ?")
        params.append(recipient_id)
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM app_notifications WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at DESC, notification_id DESC LIMIT 500",
            tuple(params),
        )]


def correction_audit(correction_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [dict(row) for row in db.execute(
            """SELECT * FROM attendance_correction_audit
               WHERE correction_id = ? ORDER BY created_at, audit_id""",
            (correction_id,),
        )]


def missing_checkout_count(department="", database_path=DATABASE_FILE):
    where = " WHERE a.missing_checkout_pending = 1"
    params = []
    if department:
        where += " AND lower(trim(e.department)) = lower(trim(?))"
        params.append(department)
    with connection(database_path) as db:
        row = db.execute(
            """SELECT COUNT(*) AS count FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id"""
            + where,
            tuple(params),
        ).fetchone()
        return row["count"]


def pending_correction_count(department="", database_path=DATABASE_FILE):
    where = " WHERE r.status = 'pending'"
    params = []
    if department:
        where += " AND lower(trim(e.department)) = lower(trim(?))"
        params.append(department)
    with connection(database_path) as db:
        row = db.execute(
            """SELECT COUNT(*) AS count FROM attendance_correction_requests r
               JOIN employees e ON e.employee_id = r.employee_id"""
            + where,
            tuple(params),
        ).fetchone()
        return row["count"]


def detect_missing_checkouts(now=None, database_path=DATABASE_FILE):
    config = settings(database_path=database_path)
    timezone = ZoneInfo(config["timezone"])
    local_now = now.astimezone(timezone) if now and now.tzinfo else (
        now.replace(tzinfo=timezone) if now else datetime.now(timezone)
    )
    marked = 0
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT a.attendance_id, a.employee_id, a.attendance_date,
                      e.full_name, e.department, s.start_time, s.end_time
               FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id
               JOIN shifts s ON s.shift_id = a.shift_id
               WHERE a.check_out IS NULL
                 AND (a.missing_checkout_pending IS NULL OR a.missing_checkout_pending = 0)"""
        ).fetchall()
        for row in rows:
            attendance_day = date.fromisoformat(row["attendance_date"])
            shift_start = datetime_time.fromisoformat(row["start_time"])
            shift_end = datetime_time.fromisoformat(row["end_time"])
            end_day = attendance_day + timedelta(days=1) if shift_end <= shift_start else attendance_day
            shift_end_at = datetime.combine(end_day, shift_end, timezone)
            if local_now < shift_end_at:
                continue
            db.execute(
                "UPDATE attendance SET missing_checkout_pending = 1 WHERE attendance_id = ?",
                (row["attendance_id"],),
            )
            title = f"Missing checkout pending for {row['attendance_date']}"
            message = "No final checkout was recorded. Submit an attendance correction request."
            for recipient_role, recipient_id in (
                ("employee", row["employee_id"]),
                ("hod", ""),
                ("hr", ""),
            ):
                existing = db.execute(
                    """SELECT notification_id FROM app_notifications
                       WHERE recipient_role = ? AND employee_id = ?
                         AND category = 'missing_checkout' AND title = ? LIMIT 1""",
                    (recipient_role, row["employee_id"], title),
                ).fetchone()
                if not existing:
                    db.execute(
                        """INSERT INTO app_notifications
                           (recipient_role, recipient_id, employee_id, category, title, message, created_at)
                           VALUES (?, ?, ?, 'missing_checkout', ?, ?, ?)""",
                        (recipient_role, recipient_id, row["employee_id"], title, message,
                         local_now.isoformat(timespec="seconds")),
                    )
            marked += 1
    return marked


def update_default_shift(start_time, end_time, grace_period_minutes, standard_work_hours,
                         database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.execute(
            """UPDATE shifts SET start_time = ?, end_time = ?,
               grace_period_minutes = ?, standard_work_hours = ? WHERE shift_id = 1""",
            (start_time, end_time, grace_period_minutes, standard_work_hours),
        )
        db.execute(
            """UPDATE app_settings SET setting_value = ?
               WHERE setting_key = 'office_start'""",
            (start_time,),
        )
        db.execute(
            """UPDATE app_settings SET setting_value = ?
               WHERE setting_key = 'office_end'""",
            (end_time,),
        )
        db.execute(
            """UPDATE app_settings SET setting_value = ?
               WHERE setting_key = 'grace_period_minutes'""",
            (str(grace_period_minutes),),
        )
        db.execute(
            """UPDATE app_settings SET setting_value = ?
               WHERE setting_key = 'standard_work_hours'""",
            (str(standard_work_hours),),
        )


def _qatar_now(now, timezone_name):
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
    if now is None:
        return datetime.now(timezone)
    return now.astimezone(timezone) if now.tzinfo else now.replace(tzinfo=timezone)


def record_attendance(employee_id, action, now=None, timezone_name="Asia/Qatar",
                      database_path=DATABASE_FILE, photo_path=None,
                      recognition_score=None, liveness_score=None):
    if action not in {"check_in", "check_out"}:
        raise ValueError("Action must be check_in or check_out.")
    local_now = _qatar_now(now, timezone_name)
    stamp = local_now.isoformat(timespec="seconds")
    day = local_now.date().isoformat()
    with connection(database_path) as db:
        employee = db.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
        if employee is None:
            raise LookupError("Employee not found.")
        if employee["status"] != "active":
            raise PermissionError("Inactive employees cannot record attendance.")
        if action == "check_in":
            shift = db.execute(
                """SELECT s.* FROM shifts s JOIN employees e ON e.shift_id = s.shift_id
                   WHERE e.employee_id = ? AND s.active = 1""",
                (employee_id,),
            ).fetchone()
            if shift is None:
                raise LookupError("No active shift is configured.")
            open_record = db.execute(
                """SELECT attendance_id FROM attendance
                   WHERE employee_id = ? AND check_out IS NULL LIMIT 1""",
                (employee_id,),
            ).fetchone()
            shift_start = datetime_time.fromisoformat(shift["start_time"])
            shift_end = datetime_time.fromisoformat(shift["end_time"])
            attendance_day = local_now.date()
            if shift_end <= shift_start and local_now.time() < shift_end:
                attendance_day -= timedelta(days=1)
            day = attendance_day.isoformat()
            existing = db.execute(
                """SELECT * FROM attendance
                   WHERE employee_id = ? AND attendance_date = ? AND shift_id = ?""",
                (employee_id, day, shift["shift_id"]),
            ).fetchone()
            if existing:
                if open_record:
                    raise FileExistsError("Employee already has an open check-in.")
                last_checkout = datetime.fromisoformat(existing["check_out"])
                if local_now <= last_checkout:
                    raise ValueError("Return check-in must be later than the previous check-out.")
                db.execute(
                    """INSERT INTO attendance_movements
                       (attendance_id, action, occurred_at, photo_path,
                        recognition_score, liveness_score)
                       VALUES (?, 'check_out', ?, ?, ?, ?)""",
                    (
                        existing["attendance_id"], existing["check_out"],
                        existing["checkout_photo_path"],
                        existing["checkout_recognition_score"],
                        existing["checkout_liveness_score"],
                    ),
                )
                db.execute(
                    """INSERT INTO attendance_movements
                       (attendance_id, action, occurred_at, photo_path,
                        recognition_score, liveness_score)
                       VALUES (?, 'check_in', ?, ?, ?, ?)""",
                    (
                        existing["attendance_id"], stamp, photo_path,
                        recognition_score, liveness_score,
                    ),
                )
                db.execute(
                    """UPDATE attendance SET check_out = NULL, working_hours = NULL,
                       early_departure = 0, checkout_photo_path = NULL,
                       checkout_photo_timestamp = NULL,
                       checkout_recognition_score = NULL,
                       checkout_liveness_score = NULL, status = 'present'
                       WHERE attendance_id = ?""",
                    (existing["attendance_id"],),
                )
                attendance_id = existing["attendance_id"]
                returned_from_break = True
            else:
                if open_record:
                    raise FileExistsError("Employee already has an open check-in.")
                returned_from_break = False
            late_after = (
                datetime.combine(attendance_day, shift_start, local_now.tzinfo)
                + timedelta(minutes=shift["grace_period_minutes"])
            )
            late_after_grace = int(local_now > late_after)
            late = late_after_grace
            if not returned_from_break:
                try:
                    cursor = db.execute(
                        """INSERT INTO attendance
                           (employee_id, shift_id, attendance_date, check_in, late,
                            late_after_grace, checkin_photo_path,
                            checkin_photo_timestamp, checkin_recognition_score,
                            checkin_liveness_score, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'present')""",
                        (
                            employee_id, shift["shift_id"], day, stamp, late,
                            late_after_grace, photo_path, stamp if photo_path else None,
                            recognition_score if photo_path else None,
                            liveness_score if photo_path else None,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise FileExistsError(
                        "Attendance has already been checked in for this shift."
                    ) from exc
                attendance_id = cursor.lastrowid
        else:
            existing = db.execute(
                """SELECT * FROM attendance WHERE employee_id = ? AND check_out IS NULL
                   ORDER BY attendance_date DESC, check_in DESC LIMIT 1""",
                (employee_id,),
            ).fetchone()
            if existing is None:
                raise LookupError("No open check-in was found for this employee.")
            if local_now <= datetime.fromisoformat(existing["check_in"]):
                raise ValueError("Check-out must be later than check-in.")
            check_in = datetime.fromisoformat(existing["check_in"])
            hours = round((local_now - check_in).total_seconds() / 3600, 2)
            attendance_shift = db.execute(
                "SELECT * FROM shifts WHERE shift_id = ?", (existing["shift_id"],)
            ).fetchone()
            shift_end = datetime_time.fromisoformat(attendance_shift["end_time"])
            shift_start = datetime_time.fromisoformat(attendance_shift["start_time"])
            end_day = datetime.fromisoformat(existing["attendance_date"]).date()
            if shift_end <= shift_start:
                end_day += timedelta(days=1)
            scheduled_end = datetime.combine(end_day, shift_end, local_now.tzinfo)
            cursor = db.execute(
                """UPDATE attendance SET check_out = ?, working_hours = ?,
                   early_departure = ?, checkout_photo_path = ?,
                   checkout_photo_timestamp = ?, checkout_recognition_score = ?,
                   checkout_liveness_score = ?,
                   status = 'checked_out'
                   WHERE attendance_id = ? AND check_out IS NULL""",
                (
                    stamp, hours, int(local_now < scheduled_end), photo_path,
                    stamp if photo_path else None,
                    recognition_score if photo_path else None,
                    liveness_score if photo_path else None,
                    existing["attendance_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise FileExistsError("This attendance record was already checked out.")
            attendance_id = existing["attendance_id"]
            returned_from_break = False
    result = attendance_record(attendance_id, database_path)
    result["attendance_action_time"] = stamp
    if action == "check_in":
        result["returned_from_break"] = returned_from_break
    reconcile_daily_attendance_summaries(database_path, day)
    return result


def attendance_photo_path(attendance_id, action, database_path=DATABASE_FILE):
    column = {
        "check_in": "checkin_photo_path",
        "check_out": "checkout_photo_path",
    }.get(action)
    if column is None:
        raise ValueError("Action must be check_in or check_out.")
    with connection(database_path) as db:
        row = db.execute(
            f"SELECT {column} FROM attendance WHERE attendance_id = ?",
            (attendance_id,),
        ).fetchone()
        return row[column] if row else None


def open_attendance_photo_path(employee_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            """SELECT checkin_photo_path FROM attendance
               WHERE employee_id = ?
               ORDER BY attendance_date DESC, check_in DESC LIMIT 1""",
            (employee_id,),
        ).fetchone()
        return row["checkin_photo_path"] if row else None


def clear_attendance_photo(attendance_id, action, database_path=DATABASE_FILE):
    columns = {
        "check_in": (
            "checkin_photo_path", "checkin_photo_timestamp",
            "checkin_recognition_score", "checkin_liveness_score",
        ),
        "check_out": (
            "checkout_photo_path", "checkout_photo_timestamp",
            "checkout_recognition_score", "checkout_liveness_score",
        ),
    }.get(action)
    if columns is None:
        raise ValueError("Action must be check_in or check_out.")
    path_column, timestamp_column, score_column, liveness_column = columns
    with connection(database_path) as db:
        row = db.execute(
            f"SELECT {path_column} FROM attendance WHERE attendance_id = ?",
            (attendance_id,),
        ).fetchone()
        if row is None:
            return None
        path = row[path_column]
        db.execute(
            f"""UPDATE attendance SET {path_column} = NULL,
               {timestamp_column} = NULL, {score_column} = NULL,
               {liveness_column} = NULL
               WHERE attendance_id = ?""",
            (attendance_id,),
        )
        return path


def attendance_photo_records(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [
            dict(row) for row in db.execute(
                """SELECT attendance_id, checkin_photo_path, checkout_photo_path,
                          checkin_photo_timestamp, checkout_photo_timestamp
                   FROM attendance
                   WHERE checkin_photo_path IS NOT NULL
                      OR checkout_photo_path IS NOT NULL"""
            )
        ]


def attendance_movement_photo_records(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [
            dict(row) for row in db.execute(
                """SELECT movement_id, occurred_at AS photo_timestamp, photo_path
                   FROM attendance_movements
                   WHERE photo_path IS NOT NULL"""
            )
        ]


def attendance_photo_paths(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [
            path
            for row in db.execute(
                """SELECT checkin_photo_path, checkout_photo_path FROM attendance"""
            )
            for path in row
            if path
        ] + [
            row["photo_path"]
            for row in db.execute(
                "SELECT photo_path FROM attendance_movements WHERE photo_path IS NOT NULL"
            )
        ]


def attendance_movements(attendance_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        return [
            dict(row) for row in db.execute(
                """SELECT movement_id, attendance_id, action, occurred_at,
                          photo_path, recognition_score, liveness_score
                   FROM attendance_movements
                   WHERE attendance_id = ?
                   ORDER BY occurred_at, movement_id""",
                (attendance_id,),
            )
        ]


def attendance_movement_photo_path(attendance_id, movement_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            """SELECT photo_path FROM attendance_movements
               WHERE attendance_id = ? AND movement_id = ?""",
            (attendance_id, movement_id),
        ).fetchone()
        return row["photo_path"] if row else None


def clear_attendance_movement_photo(movement_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            "SELECT photo_path FROM attendance_movements WHERE movement_id = ?",
            (movement_id,),
        ).fetchone()
        if row is None:
            return None
        db.execute(
            "UPDATE attendance_movements SET photo_path = NULL WHERE movement_id = ?",
            (movement_id,),
        )
        return row["photo_path"]


def attendance_record(attendance_id, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            """SELECT a.*, e.full_name, e.face_name, e.department, e.designation,
                      s.name AS shift_name, s.start_time AS shift_start,
                      s.end_time AS shift_end,
                      s.grace_period_minutes, s.on_time_window_minutes
               FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id
               JOIN shifts s ON s.shift_id = a.shift_id
               WHERE a.attendance_id = ?""",
            (attendance_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result.update(classify_attendance(result))
        return result


def reconcile_daily_attendance_summaries(database_path=DATABASE_FILE, day=""):
    timezone = ZoneInfo(settings(database_path=database_path)["timezone"])
    local_now = datetime.now(timezone)
    with connection(database_path) as db:
        date_filter = " AND a.attendance_date = ?" if day else ""
        params = (day,) if day else ()
        rows = db.execute(
            """SELECT a.attendance_id, a.employee_id, a.attendance_date,
                      a.check_in, a.check_out, a.working_hours,
                      a.status, a.late, a.late_after_grace, a.early_departure,
                      a.early_departure AS early_value,
                      a.missing_checkout_pending,
                      e.full_name, e.department, e.shift_id,
                      s.start_time AS shift_start, s.end_time AS shift_end,
                      s.grace_period_minutes, s.on_time_window_minutes
               FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id
               JOIN shifts s ON s.shift_id = a.shift_id
               WHERE 1=1""" + date_filter,
            params,
        ).fetchall()
        for row in rows:
            attendance = dict(row)
            attendance_class = classify_attendance(attendance)
            movements = db.execute(
                """SELECT action, occurred_at FROM attendance_movements
                   WHERE attendance_id = ? ORDER BY occurred_at, movement_id""",
                (attendance["attendance_id"],),
            ).fetchall()
            first_check_in = datetime.fromisoformat(attendance["check_in"])
            final_checkout = (
                datetime.fromisoformat(attendance["check_out"])
                if attendance["check_out"] else None
            )
            intervals = []
            interval_start = first_check_in
            break_seconds = 0
            pending_exit = None
            for movement in movements:
                occurred = datetime.fromisoformat(movement["occurred_at"])
                if movement["action"] == "check_out":
                    if interval_start is not None:
                        intervals.append((interval_start, occurred))
                        interval_start = None
                    pending_exit = occurred
                elif movement["action"] == "check_in" and pending_exit is not None:
                    break_seconds += max(0, (occurred - pending_exit).total_seconds())
                    interval_start = occurred
                    pending_exit = None
            if final_checkout is not None:
                if interval_start is not None:
                    intervals.append((interval_start, final_checkout))
            interval_seconds = sum(
                max(0, (end - start).total_seconds())
                for start, end in intervals
            )
            offsite_rows = db.execute(
                """SELECT departure_time, return_time FROM offsite_meeting_records
                   WHERE employee_id = ? AND attendance_date = ?
                     AND status = 'approved'""",
                (attendance["employee_id"], attendance["attendance_date"]),
            ).fetchall()
            offsite_intervals = []
            for offsite in offsite_rows:
                if not offsite["return_time"]:
                    continue
                offsite_intervals.append((
                    datetime.fromisoformat(offsite["departure_time"]),
                    datetime.fromisoformat(offsite["return_time"]),
                ))
            merged_offsite = []
            for start, end in sorted(offsite_intervals):
                if merged_offsite and start <= merged_offsite[-1][1]:
                    merged_offsite[-1] = (merged_offsite[-1][0], max(merged_offsite[-1][1], end))
                else:
                    merged_offsite.append((start, end))
            offsite_seconds = sum(
                max(0, (end - start).total_seconds())
                for start, end in merged_offsite
            )
            working_seconds = interval_seconds + offsite_seconds
            working_hours = round(working_seconds / 3600, 2) if intervals or final_checkout or merged_offsite else None
            break_hours = round(break_seconds / 3600, 2)
            offsite_hours = round(offsite_seconds / 3600, 2)
            attendance_day = date.fromisoformat(attendance["attendance_date"])
            shift_start = datetime_time.fromisoformat(attendance["shift_start"])
            shift_end = datetime_time.fromisoformat(attendance["shift_end"])
            end_day = attendance_day + timedelta(days=1) if shift_end <= shift_start else attendance_day
            shift_end_at = datetime.combine(end_day, shift_end, timezone)
            if not final_checkout and local_now >= shift_end_at:
                summary_status = "MISSING_CHECKOUT"
                verification_status = "pending_verification"
            elif final_checkout:
                summary_status = "EARLY_CHECKOUT" if attendance_class["early_departure_minutes"] > 0 else "CHECKED_OUT"
                verification_status = "verified" if attendance["status"] == "checked_out" else "pending_verification"
            elif not final_checkout:
                summary_status = "CURRENTLY_WORKING"
                verification_status = "open_check_in"
            else:
                summary_status = "ABSENT"
                verification_status = "none"
            correction_status = "none"
            correction = db.execute(
                """SELECT status FROM attendance_correction_requests
                   WHERE employee_id = ? AND attendance_date = ?
                   ORDER BY correction_id DESC LIMIT 1""",
                (attendance["employee_id"], attendance["attendance_date"]),
            ).fetchone()
            if correction:
                correction_status = correction["status"]
            punctuality_status = attendance_class["attendance_status"]
            now_text = local_now.isoformat(timespec="seconds")
            db.execute(
                """INSERT INTO daily_attendance_summary
                   (employee_id, attendance_date, first_check_in, final_checkout,
                    working_hours, break_duration, offsite_duration, attendance_status,
                    punctuality_status, checkout_verification_status,
                    correction_approval_status, early_departure_minutes,
                    late_minutes, last_reconciled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(employee_id, attendance_date) DO UPDATE SET
                    first_check_in=excluded.first_check_in,
                    final_checkout=excluded.final_checkout,
                    working_hours=excluded.working_hours,
                    break_duration=excluded.break_duration,
                    offsite_duration=excluded.offsite_duration,
                    attendance_status=excluded.attendance_status,
                    punctuality_status=excluded.punctuality_status,
                    checkout_verification_status=excluded.checkout_verification_status,
                    correction_approval_status=excluded.correction_approval_status,
                    early_departure_minutes=excluded.early_departure_minutes,
                    late_minutes=excluded.late_minutes,
                    last_reconciled_at=excluded.last_reconciled_at""",
                (
                    attendance["employee_id"], attendance["attendance_date"],
                    attendance["check_in"], attendance["check_out"], working_hours,
                    break_hours, offsite_hours, summary_status, punctuality_status,
                    verification_status, correction_status,
                    attendance_class["early_departure_minutes"],
                    attendance_class["late_minutes"], now_text,
                ),
            )
    return True


def day_wise_attendance(start_date, end_date="", department="", employee_id="",
                        status="", database_path=DATABASE_FILE):
    reconcile_daily_attendance_summaries(database_path)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date or start_date)
    current = start
    while current <= end:
        snapshot = daily_attendance(current.isoformat(), database_path=database_path)
        for row in snapshot["records"]:
            if department and (row.get("department") or "").strip().casefold() != department.strip().casefold():
                continue
            if employee_id and row["employee_id"] != employee_id:
                continue
            summary_status = {
                "ON_TIME": "PRESENT",
                "GRACE_PERIOD": "PRESENT",
                "LATE": "LATE",
                "ABSENT": "ABSENT",
                "ON_LEAVE": "ON_LEAVE",
                "HOLIDAY": "HOLIDAY",
                "WEEKLY_OFF": "HOLIDAY",
                "NOT_CHECKED_IN": "NOT_CHECKED_IN",
                "NOT_SCHEDULED": "NOT_SCHEDULED",
                "CURRENTLY_WORKING": "CURRENTLY_WORKING",
                "CHECKED_OUT": "CHECKED_OUT",
            }.get(row["status"], row["status"])
            if row.get("attendance_id"):
                continue
            with connection(database_path) as db:
                db.execute(
                    """INSERT INTO daily_attendance_summary
                       (employee_id, attendance_date, first_check_in, final_checkout,
                        working_hours, break_duration, offsite_duration, attendance_status,
                        punctuality_status, checkout_verification_status,
                        correction_approval_status, early_departure_minutes,
                        late_minutes, last_reconciled_at)
                       VALUES (?, ?, NULL, NULL, NULL, 0, 0, ?, ?, 'none', 'none', 0, 0, ?)
                       ON CONFLICT(employee_id, attendance_date) DO UPDATE SET
                        attendance_status=excluded.attendance_status,
                        punctuality_status=excluded.punctuality_status,
                        last_reconciled_at=excluded.last_reconciled_at""",
                    (
                        row["employee_id"], current.isoformat(), summary_status,
                        row.get("attendance_status") or row["status"],
                        datetime.now(ZoneInfo(settings(database_path=database_path)["timezone"])).isoformat(timespec="seconds"),
                    ),
                )
        current += timedelta(days=1)
    clauses = ["d.attendance_date BETWEEN ? AND ?"]
    params = [start_date, end_date or start_date]
    if department:
        clauses.append("lower(trim(e.department)) = lower(trim(?))")
        params.append(department)
    if employee_id:
        clauses.append("d.employee_id = ?")
        params.append(employee_id)
    if status:
        clauses.append("d.attendance_status = ?")
        params.append(status)
    with connection(database_path) as db:
        rows = db.execute(
            """SELECT d.*, e.full_name, e.department, e.designation
               FROM daily_attendance_summary d
               JOIN employees e ON e.employee_id = d.employee_id
               WHERE """ + " AND ".join(clauses)
            + " ORDER BY d.attendance_date DESC, e.full_name COLLATE NOCASE",
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]


def attendance_record_by_employee_date(employee_id, attendance_date, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            """SELECT * FROM attendance
               WHERE employee_id = ? AND attendance_date = ?
               ORDER BY attendance_id LIMIT 1""",
            (employee_id, attendance_date),
        ).fetchone()
        return dict(row) if row else None


def attendance_records(start_date="", end_date="", name="", department="",
                       database_path=DATABASE_FILE, employee_id=""):
    clauses, params = [], []
    if start_date:
        clauses.append("a.attendance_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("a.attendance_date <= ?")
        params.append(end_date)
    if name:
        clauses.append("e.full_name LIKE ?")
        params.append(f"%{name}%")
    if department:
        clauses.append("e.department = ?")
        params.append(department)
    if employee_id:
        clauses.append("a.employee_id = ?")
        params.append(employee_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connection(database_path) as db:
        rows = db.execute(
            f"""SELECT a.*, e.full_name, e.department, e.designation, s.name AS shift_name,
                       s.start_time AS shift_start, s.end_time AS shift_end,
                       s.grace_period_minutes, s.on_time_window_minutes
                FROM attendance a JOIN employees e ON e.employee_id = a.employee_id
                JOIN shifts s ON s.shift_id = a.shift_id {where}
                ORDER BY a.attendance_date DESC, a.check_in DESC LIMIT 10000""",
            params,
        )
        result = [dict(row) for row in rows]
        for record in result:
            record.update(classify_attendance(record))
            record["status_label"] = record["attendance_status"].replace("_", " ")
        return result


def legacy_attendance_records(start_date="", end_date="", name="",
                              database_path=DATABASE_FILE, employee_id=""):
    clauses = [
        "ae.event = 'attendance'",
        "ae.status IN ('checked_in', 'checked_out')",
        """ae.event_id > COALESCE(
               (SELECT CAST(setting_value AS INTEGER) FROM app_settings
                WHERE setting_key = ?), 0)""",
    ]
    params = [LEGACY_ATTENDANCE_CLEAR_EVENT_ID_SETTING]
    if start_date:
        clauses.append("substr(ae.time, 1, 10) >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("substr(ae.time, 1, 10) <= ?")
        params.append(end_date)
    if name:
        clauses.append("e.full_name LIKE ?")
        params.append(f"%{name}%")
    if employee_id:
        clauses.append("e.employee_id = ?")
        params.append(employee_id)
    where = " AND ".join(clauses)
    with connection(database_path) as db:
        events = db.execute(
            f"""SELECT ae.event_id, ae.time, ae.status AS event_status,
                       ae.name AS face_name, e.employee_id, e.full_name, e.department
                FROM access_events ae
                JOIN employees e ON e.face_name = ae.name
                WHERE {where}
                ORDER BY ae.time, ae.event_id""",
            params,
        ).fetchall()

    timezone = ZoneInfo(settings(database_path=database_path)["timezone"])
    records = []
    open_records = {}
    for event in events:
        item = dict(event)
        event_time = datetime.fromisoformat(item["time"])
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone)
        item["time"] = event_time.isoformat(timespec="seconds")
        day = event_time.date().isoformat()
        key = (item["employee_id"], day)
        if item["event_status"] == "checked_in":
            record = {
                "employee_id": item["employee_id"],
                "full_name": item["full_name"],
                "department": item["department"],
                "attendance_date": day,
                "check_in": item["time"],
                "check_out": None,
                "working_hours": None,
                "late": None,
                "early_departure": None,
                "status": "present",
                "legacy_event": True,
                "legacy_event_id": item["event_id"],
            }
            records.append(record)
            open_records.setdefault(key, []).append(record)
        else:
            candidates = open_records.get(key, [])
            check_out = event_time
            match = next(
                (
                    candidate for candidate in reversed(candidates)
                    if check_out > datetime.fromisoformat(candidate["check_in"])
                ),
                None,
            )
            if match is None:
                continue
            candidates.remove(match)
            record = match
            record["check_out"] = item["time"]
            record["working_hours"] = round(
                (
                    check_out - datetime.fromisoformat(record["check_in"])
                ).total_seconds() / 3600,
                2,
            )
            record["status"] = "checked_out"
    return records


def clear_attendance_records(database_path=DATABASE_FILE):
    with connection(database_path) as db:
        cursor = db.execute("DELETE FROM attendance")
        latest_event_id = db.execute(
            "SELECT COALESCE(MAX(event_id), 0) FROM access_events"
        ).fetchone()[0]
        db.execute(
            """INSERT INTO app_settings (setting_key, setting_value)
               VALUES (?, ?)
               ON CONFLICT(setting_key) DO UPDATE SET
                   setting_value = excluded.setting_value""",
            (
                LEGACY_ATTENDANCE_CLEAR_EVENT_ID_SETTING,
                str(latest_event_id),
            ),
        )
        return cursor.rowcount


def dashboard_stats(today, database_path=DATABASE_FILE):
    snapshot = daily_attendance(today, database_path=database_path)
    counts = snapshot["counts"]
    return {
        "total_employees": snapshot["total_scheduled"],
        "present_today": sum(
            counts[key] for key in
            ("ON_TIME", "GRACE_PERIOD", "LATE", "CURRENTLY_WORKING", "CHECKED_OUT")
        ),
        "absent_today": counts["ABSENT"],
        "late_arrivals": snapshot["punctuality_counts"]["LATE"],
        "currently_working": counts["CURRENTLY_WORKING"],
        "checked_out": counts["CHECKED_OUT"],
    }


def daily_attendance(day, now=None, database_path=DATABASE_FILE, employee_id=""):
    selected_date = date.fromisoformat(day) if isinstance(day, str) else day
    config = settings(database_path=database_path)
    timezone = ZoneInfo(config["timezone"])
    local_now = _qatar_now(now, config["timezone"])
    weekdays = _working_days(config)
    holidays = _holiday_dates(config)
    with connection(database_path) as db:
        employee_filter = " AND e.employee_id = ?" if employee_id else ""
        employees_query = db.execute(
            """SELECT e.employee_id, e.face_name, e.full_name, e.department,
                      e.status AS employee_status, e.shift_id,
                      s.name AS shift_name, s.start_time AS shift_start,
                      s.end_time AS shift_end, s.grace_period_minutes,
                      s.on_time_window_minutes,
                      s.active AS shift_active,
                      EXISTS (
                          SELECT 1 FROM leave_requests l
                          WHERE l.employee_id = e.employee_id AND l.status = 'approved'
                            AND l.start_date <= ? AND l.end_date >= ?
                      ) AS approved_leave
               FROM employees e LEFT JOIN shifts s ON s.shift_id = e.shift_id
               WHERE e.status = 'active'""" + employee_filter +
            " ORDER BY e.full_name COLLATE NOCASE",
            (
                selected_date.isoformat(), selected_date.isoformat(),
                *((employee_id,) if employee_id else ()),
            ),
        ).fetchall()
        attendance_employee_filter = " AND a.employee_id = ?" if employee_id else ""
        records = db.execute(
            """SELECT a.*, e.full_name, e.face_name, e.department,
                      s.name AS shift_name, s.start_time AS shift_start,
                      s.end_time AS shift_end, s.grace_period_minutes,
                      s.on_time_window_minutes
               FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id
               JOIN shifts s ON s.shift_id = a.shift_id
               WHERE (a.attendance_date = ?
                  OR (a.attendance_date = ? AND a.check_out IS NULL
                      AND s.end_time <= s.start_time))""" +
            attendance_employee_filter +
            " ORDER BY a.check_in",
            (
                selected_date.isoformat(),
                (selected_date - timedelta(days=1)).isoformat(),
                *((employee_id,) if employee_id else ()),
            ),
        ).fetchall()
    by_employee = {row["employee_id"]: dict(row) for row in records}
    day_name = WEEKDAYS[selected_date.weekday()]
    day_kind = (
        "HOLIDAY" if selected_date in holidays else
        "WEEKLY_OFF" if day_name not in weekdays else None
    )
    results = []
    counts = {status: 0 for status in ATTENDANCE_STATUSES}
    punctuality_counts = {
        status: 0 for status in ("ON_TIME", "GRACE_PERIOD", "LATE", "ABSENT")
    }
    for row in employees_query:
        employee = dict(row)
        record = by_employee.get(employee["employee_id"])
        if record:
            record.update(classify_attendance(record))
            if not record["check_out"] and selected_date == local_now.date():
                status = "CURRENTLY_WORKING"
            elif record["check_out"]:
                status = "CHECKED_OUT"
            else:
                status = record["attendance_status"]
            result = {**employee, **record, "status": status}
        elif employee["approved_leave"]:
            status = "ON_LEAVE"
            result = {**employee, "status": status, "check_in": None, "check_out": None}
        elif day_kind:
            status = day_kind
            result = {**employee, "status": status, "check_in": None, "check_out": None}
        elif not employee["shift_id"] or not employee["shift_active"]:
            status = "NOT_SCHEDULED"
            result = {**employee, "status": status, "check_in": None, "check_out": None}
        else:
            shift_start = datetime_time.fromisoformat(employee["shift_start"])
            deadline = datetime.combine(
                selected_date, shift_start, timezone
            ) + timedelta(minutes=employee["grace_period_minutes"])
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone)
            if selected_date > local_now.date() or (
                selected_date == local_now.date() and local_now <= deadline
            ):
                status = "NOT_CHECKED_IN"
            else:
                status = "ABSENT"
            result = {**employee, "status": status, "check_in": None, "check_out": None}
        counts[status] += 1
        if record:
            punctuality_counts[record["attendance_status"]] += 1
        elif status == "ABSENT":
            punctuality_counts["ABSENT"] += 1
        results.append(result)
    scheduled_statuses = {
        "ON_TIME", "GRACE_PERIOD", "LATE", "CURRENTLY_WORKING", "CHECKED_OUT", "ABSENT"
    }
    total_scheduled = sum(
        counts[status] for status in scheduled_statuses | {"NOT_CHECKED_IN"}
    )
    punctuality_denominator = sum(punctuality_counts.values())
    percentages = {
        key: round(100 * punctuality_counts[key] / punctuality_denominator, 1)
        if punctuality_denominator else 0
        for key in ("ON_TIME", "GRACE_PERIOD", "LATE", "ABSENT")
    }
    return {
        "date": selected_date.isoformat(),
        "records": results,
        "counts": counts,
        "punctuality_counts": punctuality_counts,
        "total_scheduled": total_scheduled,
        "percentages": percentages,
        "day_type": day_kind,
    }


def analytics(start_date, end_date, database_path=DATABASE_FILE):
    rows = attendance_records(start_date, end_date, database_path=database_path)
    daily = {}
    departments = {}
    for row in rows:
        day = daily.setdefault(
            row["attendance_date"],
            {
                "attendance_date": row["attendance_date"],
                "present": 0,
                "late": 0,
                "early_departures": 0,
                "average_hours": None,
            },
        )
        day["present"] += 1
        day["late"] += row["late"]
        day["early_departures"] += int(bool(row["early_departure"]))
        department = departments.setdefault(
            row["department"] or "",
            {
                "department": row["department"] or "",
                "present": 0,
                "early_departures": 0,
            },
        )
        department["present"] += 1
        department["early_departures"] += int(bool(row["early_departure"]))
        if row["working_hours"] is not None:
            day["hours_total"] = day.get("hours_total", 0) + row["working_hours"]
            day["hours_count"] = day.get("hours_count", 0) + 1
    for day in daily.values():
        day["average_hours"] = (
            round(day.pop("hours_total") / day.pop("hours_count"), 2)
            if day.get("hours_count") else None
        )
        day.pop("hours_total", None)
        day.pop("hours_count", None)
    return {
        "daily": [daily[key] for key in sorted(daily)],
        "departments": [departments[key] for key in sorted(departments)],
    }


def chatbot_history(session_key, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        messages = [
            dict(row)
            for row in db.execute(
                """SELECT role, content, photos_json, created_at FROM chatbot_messages
                   WHERE session_key = ? ORDER BY message_id DESC LIMIT 100""",
                (session_key,),
            )
        ][::-1]
        for message in messages:
            message["photos"] = json.loads(message.pop("photos_json") or "[]")
        return messages


def chatbot_context(session_key, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        row = db.execute(
            "SELECT context_json FROM chatbot_context WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        return json.loads(row["context_json"]) if row else {}


def save_chatbot_turn(
    session_key, question, answer, context, database_path=DATABASE_FILE, photos=None
):
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection(database_path) as db:
        db.execute(
            """INSERT INTO chatbot_messages(session_key, role, content, created_at)
               VALUES (?, 'user', ?, ?)""",
            (session_key, question, created_at),
        )
        db.execute(
            """INSERT INTO chatbot_messages
               (session_key, role, content, photos_json, created_at)
               VALUES (?, 'assistant', ?, ?, ?)""",
            (session_key, answer, json.dumps(photos or []), created_at),
        )
        db.execute(
            """INSERT INTO chatbot_context(session_key, context_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_key) DO UPDATE SET
                   context_json = excluded.context_json,
                   updated_at = excluded.updated_at""",
            (session_key, json.dumps(context), created_at),
        )
        db.execute(
            """DELETE FROM chatbot_messages WHERE session_key = ? AND message_id NOT IN
               (SELECT message_id FROM chatbot_messages WHERE session_key = ?
                ORDER BY message_id DESC LIMIT 100)""",
            (session_key, session_key),
        )


def clear_chatbot_history(session_key, database_path=DATABASE_FILE):
    with connection(database_path) as db:
        db.execute("DELETE FROM chatbot_messages WHERE session_key = ?", (session_key,))
        db.execute("DELETE FROM chatbot_context WHERE session_key = ?", (session_key,))
