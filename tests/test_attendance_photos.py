import base64
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from fastapi import HTTPException

import database
import server


class AttendancePhotoTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "attendance.db")
        self.photo_dir = os.path.join(self.temp_dir.name, "photos")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff({"Alice": []}, {}, self.database_path)
        employee = database.employee_by_face_name("Alice", self.database_path)
        database.update_employee(
            employee["employee_id"],
            {"consent_at": "2026-01-01T08:00:00+03:00"},
            self.database_path,
        )
        self.employee = database.employee_by_id(employee["employee_id"], self.database_path)
        self.stack = []

        def db_call(function, positional_path=False):
            def call(*args, **kwargs):
                if positional_path:
                    return function(args[0], self.database_path)
                kwargs["database_path"] = self.database_path
                return function(*args, **kwargs)
            return call

        for name in (
            "employee_by_face_name",
            "settings",
            "record_attendance",
            "open_attendance_photo_path",
            "attendance_photo_path",
            "clear_attendance_photo",
            "attendance_record",
        ):
            item = patch.object(
                server.attendance_db,
                name,
                db_call(
                    getattr(database, name),
                    positional_path=name == "attendance_record",
                ),
            )
            item.start()
            self.stack.append(item)
        for name, value in (
            ("ATTENDANCE_PHOTO_DIR", self.photo_dir),
            ("detect", lambda image: [[0, 0, 12, 12]]),
            ("embed", lambda image, face: np.zeros(1, dtype=np.float32)),
            ("identify", lambda feature: ("Alice", 0.95)),
            ("assess_liveness", lambda image, face: 0.91),
            ("append_entry", lambda *args, **kwargs: None),
            ("say", lambda *args, **kwargs: None),
        ):
            item = patch.object(server, name, value)
            item.start()
            self.stack.append(item)
        server.RECOGNITION_PROOFS.clear()

    def tearDown(self):
        server.RECOGNITION_PROOFS.clear()
        for item in reversed(self.stack):
            item.stop()
        self.temp_dir.cleanup()

    def image_data(self, shade):
        ok, image = cv2.imencode(
            ".jpg", np.full((48, 64, 3), shade, dtype=np.uint8)
        )
        self.assertTrue(ok)
        return base64.b64encode(image.tobytes()).decode("ascii")

    def attendance_data(self, action, proof_prefix, photo_data, at=None):
        recognized_at = at or datetime.now(ZoneInfo("Asia/Qatar"))
        proof_names = [f"{proof_prefix}-{index}" for index in range(3)]
        expires = time.time() + 120
        for index, token in enumerate(proof_names):
            server.RECOGNITION_PROOFS[token] = (
                "Alice", 0.95, expires, recognized_at.timestamp() + index * 0.01,
            )
        return server.AttendanceClock(
            name="Alice",
            action=action,
            proofs=proof_names,
            photo_data=photo_data,
        )

    def test_face_checkin_and_checkout_store_distinct_live_photos(self):
        checkin = server.attendance_clock(
            self.attendance_data("check_in", "in", self.image_data(40))
        )
        checkout_time = datetime.fromisoformat(
            checkin["attendance"]["check_in"]
        ) + timedelta(minutes=5)
        checkout = server.attendance_clock(
            self.attendance_data(
                "check_out", "out", self.image_data(180), checkout_time
            )
        )

        response_record = checkout["attendance"]
        self.assertEqual(
            checkin["attendance"]["attendance_id"], response_record["attendance_id"]
        )
        self.assertTrue(checkin["attendance"]["checkin_photo_available"])
        self.assertTrue(response_record["checkout_photo_available"])
        self.assertNotIn("checkin_photo_path", response_record)
        record = database.attendance_record(
            response_record["attendance_id"], self.database_path
        )
        self.assertTrue(record["checkin_photo_path"].startswith("checkin/"))
        self.assertTrue(record["checkout_photo_path"].startswith("checkout/"))
        self.assertNotEqual(record["checkin_photo_path"], record["checkout_photo_path"])
        checkin_path = server.safe_attendance_photo_path(record["checkin_photo_path"])
        checkout_path = server.safe_attendance_photo_path(record["checkout_photo_path"])
        with open(checkin_path, "rb") as file:
            checkin_bytes = file.read()
        with open(checkout_path, "rb") as file:
            checkout_bytes = file.read()
        self.assertNotEqual(checkin_bytes, checkout_bytes)
        self.assertEqual(record["checkin_photo_timestamp"], record["check_in"])
        self.assertEqual(record["checkout_photo_timestamp"], record["check_out"])
        self.assertEqual(record["checkin_recognition_score"], 0.95)
        self.assertEqual(record["checkout_recognition_score"], 0.95)
        self.assertEqual(record["checkin_liveness_score"], 0.91)
        self.assertEqual(record["checkout_liveness_score"], 0.91)
        self.assertEqual(
            len(database.attendance_records(database_path=self.database_path)), 1
        )

    def test_invalid_photo_rejects_attendance_without_saving_a_record(self):
        data = self.attendance_data("check_in", "invalid", "not-base64")
        with self.assertRaises(HTTPException) as error:
            server.attendance_clock(data)

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )
        self.assertFalse(os.path.exists(self.photo_dir))

    def test_spoofed_attendance_photo_rejects_live_recognition_proofs(self):
        data = self.attendance_data("check_in", "spoof-photo", self.image_data(75))
        with patch.object(server, "assess_liveness", return_value=0.49):
            with self.assertRaises(HTTPException) as error:
                server.attendance_clock(data)

        self.assertEqual(error.exception.status_code, 403)
        self.assertIn("attendance photo", error.exception.detail)
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )
        self.assertFalse(os.path.exists(self.photo_dir))

    def test_photo_storage_failure_rejects_attendance(self):
        data = self.attendance_data("check_in", "storage-failure", self.image_data(75))
        with patch.object(server, "save_attendance_photo", side_effect=OSError("disk full")):
            with self.assertRaises(HTTPException) as error:
                server.attendance_clock(data)

        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("attendance was not recorded", error.exception.detail)
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )

    def test_missing_photo_from_stale_client_has_clear_error(self):
        data = server.AttendanceClock(
            name="Alice",
            action="check_in",
            proofs=["one", "two", "three"],
            photo_data=None,
        )
        with self.assertRaises(HTTPException) as error:
            server.attendance_clock(data)

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            error.exception.detail,
            "Live attendance photo is missing. Refresh the page and try again.",
        )
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )

    def test_duplicate_checkin_does_not_leave_an_extra_photo(self):
        first = server.attendance_clock(
            self.attendance_data("check_in", "first", self.image_data(50))
        )
        with self.assertRaises(HTTPException) as error:
            server.attendance_clock(
                self.attendance_data("check_in", "duplicate", self.image_data(120))
            )

        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(
            len(database.attendance_records(database_path=self.database_path)), 1
        )
        paths = []
        for directory, _, filenames in os.walk(self.photo_dir):
            paths.extend(os.path.join(directory, name) for name in filenames)
        self.assertEqual(
            [os.path.realpath(path) for path in paths],
            [server.safe_attendance_photo_path(
                database.attendance_record(
                    first["attendance"]["attendance_id"], self.database_path
                )["checkin_photo_path"]
            )],
        )

    def test_checkout_rejects_reuse_of_checkin_photo(self):
        photo_data = self.image_data(65)
        checkin = server.attendance_clock(
            self.attendance_data("check_in", "same-image-in", photo_data)
        )
        checkout_time = datetime.fromisoformat(
            checkin["attendance"]["check_in"]
        ) + timedelta(minutes=5)

        with self.assertRaises(HTTPException) as error:
            server.attendance_clock(
                self.attendance_data(
                    "check_out", "same-image-out", photo_data, checkout_time
                )
            )

        self.assertEqual(error.exception.status_code, 400)
        record = database.attendance_record(
            checkin["attendance"]["attendance_id"], self.database_path
        )
        self.assertIsNone(record["check_out"])
        self.assertIsNone(record["checkout_photo_path"])

    def test_unknown_employee_cannot_record_attendance_or_save_photo(self):
        with patch.object(server.attendance_db, "employee_by_face_name", return_value=None):
            with self.assertRaises(HTTPException) as error:
                server.attendance_clock(
                    self.attendance_data("check_in", "unknown", self.image_data(80))
                )

        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )
        self.assertFalse(os.path.exists(self.photo_dir))

    def test_attendance_photo_endpoint_requires_admin_authentication(self):
        qatar = ZoneInfo("Asia/Qatar")
        record = database.record_attendance(
            self.employee["employee_id"],
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="checkin/2026/01/photo.jpg",
        )
        path = server.safe_attendance_photo_path(record["checkin_photo_path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(base64.b64decode(self.image_data(90)))

        request = SimpleNamespace(cookies={})
        with self.assertRaises(HTTPException) as error:
            server.attendance_photo(
                record["attendance_id"], "check_in", request
            )
        self.assertEqual(error.exception.status_code, 401)

        token = "test-admin-session"
        server.ADMIN_SESSIONS[token] = time.time() + 60
        authorized_request = SimpleNamespace(
            cookies={server.SESSION_COOKIE: token}
        )
        response = server.attendance_photo(
            record["attendance_id"], "check_in", authorized_request
        )
        self.assertEqual(os.path.realpath(response.path), os.path.realpath(path))
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertIn("inline", response.headers["content-disposition"])
        server.ADMIN_SESSIONS.pop(token, None)

    def test_admin_can_delete_evidence_without_deleting_attendance(self):
        qatar = ZoneInfo("Asia/Qatar")
        record = database.record_attendance(
            self.employee["employee_id"],
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="checkin/2026/01/photo.jpg",
        )
        path = server.safe_attendance_photo_path(record["checkin_photo_path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(base64.b64decode(self.image_data(100)))
        token = "test-admin-session"
        server.ADMIN_SESSIONS[token] = time.time() + 60
        request = SimpleNamespace(cookies={server.SESSION_COOKIE: token})

        result = server.delete_attendance_photo(
            record["attendance_id"], "check_in", request
        )

        self.assertTrue(result["ok"])
        self.assertFalse(os.path.exists(path))
        remaining = database.attendance_record(
            record["attendance_id"], self.database_path
        )
        self.assertIsNone(remaining["checkin_photo_path"])
        self.assertEqual(
            len(database.attendance_records(database_path=self.database_path)), 1
        )
        server.ADMIN_SESSIONS.pop(token, None)
