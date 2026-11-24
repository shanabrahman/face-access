import inspect
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import datetime
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi import HTTPException, Response

import database
import server


class EmployeeAccountTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "attendance.db")
        self.photo_dir = os.path.join(self.temp_dir.name, "photos")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff({"Alice": [], "Bob": []}, {}, self.database_path)
        self.employees = {
            employee["face_name"]: employee
            for employee in database.employees(database_path=self.database_path)
        }
        self.alice = self.employees["Alice"]
        self.bob = self.employees["Bob"]
        database.update_employee(
            self.alice["employee_id"],
            {"email": "alice@example.test", "department": "Operations"},
            self.database_path,
        )
        database.update_employee(
            self.bob["employee_id"],
            {"email": "bob@example.test", "department": "Finance"},
            self.database_path,
        )
        salt, password_hash = server.employee_password_hash("alice-password-123")
        database.create_employee_account(
            self.alice["employee_id"], salt, password_hash, 24,
            self.database_path,
        )
        self.stack = ExitStack()
        for name in (
            "employee_by_id",
            "employee_account",
            "employee_account_for_login",
            "employee_account_list",
            "employees",
            "update_employee_last_login",
            "create_employee_account",
            "update_employee_account",
            "attendance_record",
            "attendance_photo_path",
            "attendance_movement_photo_path",
            "attendance_correction_requests",
            "leave_requests",
            "leave_balance",
            "settings",
            "daily_attendance",
            "legacy_attendance_records",
            "create_leave_request",
            "create_attendance_correction_request",
            "review_attendance_correction",
        ):
            function = getattr(database, name)
            parameters = list(inspect.signature(function).parameters)
            database_index = parameters.index("database_path") if "database_path" in parameters else None

            def bound(
                *args, _function=function, _database_index=database_index, **kwargs
            ):
                if (
                    "database_path" not in kwargs
                    and (_database_index is None or len(args) <= _database_index)
                ):
                    kwargs["database_path"] = self.database_path
                return _function(*args, **kwargs)

            self.stack.enter_context(patch.object(server.attendance_db, name, bound))
        self.stack.enter_context(patch.object(server, "ATTENDANCE_PHOTO_DIR", self.photo_dir))
        server.EMPLOYEE_SESSIONS.clear()
        server.EMPLOYEE_LOGIN_FAILURES.clear()
        server.EMPLOYEE_LOGIN_LOCKOUTS.clear()
        server.ADMIN_SESSIONS.clear()

    def tearDown(self):
        server.EMPLOYEE_SESSIONS.clear()
        server.ADMIN_SESSIONS.clear()
        self.stack.close()
        self.temp_dir.cleanup()

    def admin_request(self):
        token = "test-admin-session"
        server.ADMIN_SESSIONS[token] = time.time() + 60
        return SimpleNamespace(
            cookies={server.SESSION_COOKIE: token},
            client=SimpleNamespace(host="127.0.0.1"),
        )

    def employee_request(self, identifier="alice@example.test"):
        response = Response()
        server.employee_login(
            server.EmployeeLoginPayload(
                identifier=identifier, password="alice-password-123"
            ),
            SimpleNamespace(
                cookies={},
                client=SimpleNamespace(host="127.0.0.1"),
                url=SimpleNamespace(scheme="http"),
            ),
            response,
        )
        cookie = SimpleCookie()
        for key, value in response.raw_headers:
            if key.lower() == b"set-cookie":
                cookie.load(value.decode())
        return SimpleNamespace(
            cookies={server.EMPLOYEE_SESSION_COOKIE: cookie[server.EMPLOYEE_SESSION_COOKIE].value}
        )

    def test_account_creation_reuses_employee_record_and_does_not_expose_hashes(self):
        request = self.admin_request()
        server.create_employee_account(
            server.EmployeeAccountCreate(
                employee_id=self.bob["employee_id"],
                password="bob-password-123",
                annual_leave_days=18,
            ),
            request,
        )
        with self.assertRaises(HTTPException) as error:
            server.create_employee_account(
                server.EmployeeAccountCreate(
                    employee_id=self.bob["employee_id"],
                    password="bob-password-456",
                ),
                request,
            )
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(len(database.employees(database_path=self.database_path)), 2)
        account_rows = server.employee_account_list(request)
        self.assertEqual(len(account_rows), 2)
        self.assertNotIn("password_hash", account_rows[0])
        self.assertNotIn("password_salt", account_rows[0])

    def test_employee_email_login_is_scoped_to_own_profile_and_hr_sees_all(self):
        request = self.employee_request()
        profile = server.employee_profile(request)
        self.assertEqual(profile["employee_id"], self.alice["employee_id"])
        self.assertEqual(profile["department"], "Operations")
        self.assertNotIn("password_hash", profile)

        with self.assertRaises(HTTPException) as error:
            server.employee_profile(
                SimpleNamespace(cookies={server.EMPLOYEE_SESSION_COOKIE: "invalid"})
            )
        self.assertEqual(error.exception.status_code, 401)
        with self.assertRaises(HTTPException) as admin_error:
            server.employee_account_list(request)
        self.assertEqual(admin_error.exception.status_code, 401)

        employees = server.employee_list(self.admin_request())
        self.assertEqual(
            {item["employee_id"] for item in employees},
            {self.alice["employee_id"], self.bob["employee_id"]},
        )

    def test_employee_cannot_fetch_another_employees_attendance_photo(self):
        qatar = ZoneInfo("Asia/Qatar")
        alice_record = database.record_attendance(
            self.alice["employee_id"],
            "check_in",
            datetime(2026, 11, 18, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="alice/checkin.jpg",
        )
        bob_record = database.record_attendance(
            self.bob["employee_id"],
            "check_in",
            datetime(2026, 11, 18, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="bob/checkin.jpg",
        )
        for relative in ("alice/checkin.jpg", "bob/checkin.jpg"):
            path = server.safe_attendance_photo_path(relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as photo:
                photo.write(b"image")

        request = self.employee_request()
        with self.assertRaises(HTTPException) as error:
            server.employee_attendance_photo(
                bob_record["attendance_id"], "check_in", request
            )
        self.assertEqual(error.exception.status_code, 404)
        photo = server.employee_attendance_photo(
            alice_record["attendance_id"], "check_in", request
        )
        self.assertEqual(os.path.realpath(photo.path), os.path.realpath(
            server.safe_attendance_photo_path("alice/checkin.jpg")
        ))

    def test_deactivated_account_invalidates_its_employee_session(self):
        request = self.employee_request()
        server.update_employee_account(
            self.alice["employee_id"],
            server.EmployeeAccountUpdate(is_active=False),
            self.admin_request(),
        )
        with self.assertRaises(HTTPException) as error:
            server.employee_profile(request)
        self.assertEqual(error.exception.status_code, 401)
        with self.assertRaises(HTTPException) as login_error:
            server.employee_login(
                server.EmployeeLoginPayload(
                    identifier=self.alice["employee_id"],
                    password="alice-password-123",
                ),
                SimpleNamespace(
                    cookies={},
                    client=SimpleNamespace(host="127.0.0.1"),
                    url=SimpleNamespace(scheme="http"),
                ),
                Response(),
            )
        self.assertEqual(login_error.exception.status_code, 401)

    def test_employee_logout_and_password_change_revoke_sessions(self):
        request = self.employee_request()
        token = request.cookies[server.EMPLOYEE_SESSION_COOKIE]
        server.employee_logout(request, Response())
        self.assertNotIn(token, server.EMPLOYEE_SESSIONS)
        with self.assertRaises(HTTPException):
            server.employee_profile(request)

        request = self.employee_request()
        result = server.change_employee_password(
            server.EmployeePasswordChange(
                current_password="alice-password-123",
                new_password="alice-password-456",
            ),
            request,
            Response(),
        )
        self.assertTrue(result["reauthenticate"])
        with self.assertRaises(HTTPException):
            server.employee_profile(request)
        with self.assertRaises(HTTPException) as error:
            server.employee_login(
                server.EmployeeLoginPayload(
                    identifier=self.alice["employee_id"],
                    password="alice-password-123",
                ),
                SimpleNamespace(
                    cookies={},
                    client=SimpleNamespace(host="127.0.0.1"),
                    url=SimpleNamespace(scheme="http"),
                ),
                Response(),
            )
        self.assertEqual(error.exception.status_code, 401)
        new_request = SimpleNamespace(
            cookies={},
            client=SimpleNamespace(host="127.0.0.1"),
            url=SimpleNamespace(scheme="http"),
        )
        response = Response()
        server.employee_login(
            server.EmployeeLoginPayload(
                identifier=self.alice["employee_id"],
                password="alice-password-456",
            ),
            new_request,
            response,
        )
        self.assertIn("set-cookie", response.headers)

    def test_employee_attendance_history_uses_the_authenticated_employee_id(self):
        qatar = ZoneInfo("Asia/Qatar")
        for employee_id in (self.alice["employee_id"], self.bob["employee_id"]):
            database.record_attendance(
                employee_id,
                "check_in",
                datetime(2026, 11, 18, 9, 0, tzinfo=qatar),
                database_path=self.database_path,
            )
        request = self.employee_request()
        data = server.employee_attendance(request, month="2026-11")
        self.assertEqual(len(data["records"]), 30)
        self.assertTrue(
            all(item["employee_id"] == self.alice["employee_id"] for item in data["records"])
        )
        self.assertEqual(data["summary"]["present_days"], 1)
        export = server.export_employee_attendance(request, month="2026-11")
        self.assertIn("2026-11-18", export.body.decode("utf-8"))
        self.assertNotIn(self.bob["employee_id"], export.body.decode("utf-8"))

    def test_leave_balance_and_correction_requests_are_database_backed(self):
        leave_id = database.create_leave_request(
            self.alice["employee_id"], "2026-11-02", "2026-11-04", "Personal leave",
            self.database_path,
        )
        self.assertTrue(database.update_leave_status(leave_id, "approved", self.database_path))
        balance = database.leave_balance(
            self.alice["employee_id"], 2026, self.database_path
        )
        self.assertEqual(balance, {
            "year": 2026,
            "entitled_days": 24.0,
            "used_days": 3,
            "remaining_days": 21.0,
        })
        correction_id = database.create_attendance_correction_request(
            self.alice["employee_id"],
            "2026-11-03",
            "09:10",
            None,
            "Clock did not record entry",
            self.database_path,
        )
        self.assertEqual(
            database.attendance_correction_requests(
                self.database_path, self.alice["employee_id"]
            )[0]["correction_id"],
            correction_id,
        )
        self.assertTrue(database.review_attendance_correction(
            correction_id, "approved", "Reviewed", self.database_path
        ))
        self.assertEqual(
            database.attendance_correction_requests(
                self.database_path, self.alice["employee_id"]
            )[0]["reviewer_note"],
            "Reviewed",
        )
        corrected = database.attendance_records(
            employee_id=self.alice["employee_id"],
            database_path=self.database_path,
        )
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["check_in"][11:16], "09:10")


if __name__ == "__main__":
    unittest.main()
