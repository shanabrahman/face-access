import inspect
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Response

import database
import server


class HODAccountTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "attendance.db")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff({"Alice": [], "Bob": []}, {}, self.database_path)
        employees = {item["face_name"]: item for item in database.employees(database_path=self.database_path)}
        self.alice = employees["Alice"]
        self.bob = employees["Bob"]
        database.update_employee(
            self.alice["employee_id"], {"department": "Operations"}, self.database_path
        )
        database.update_employee(
            self.bob["employee_id"], {"department": "Finance"}, self.database_path
        )
        salt, password_hash = server.employee_password_hash("hod-password-123")
        self.hod_id = database.create_hod_account(
            "Alex HOD", "alex@example.test", "Operations", salt, password_hash,
            self.database_path,
        )
        self.stack = ExitStack()
        for name in (
            "employees", "employee_by_id", "hod_account_list", "hod_account_for_login",
            "hod_account_by_id", "create_hod_account", "update_hod_account",
            "delete_hod_account", "update_hod_last_login", "settings", "daily_attendance",
            "leave_requests", "leave_balance", "attendance_records", "attendance_record",
            "attendance_photo_path",
        ):
            function = getattr(database, name)
            parameters = list(inspect.signature(function).parameters)
            database_index = parameters.index("database_path") if "database_path" in parameters else None

            def bound(*args, _function=function, _database_index=database_index, **kwargs):
                if "database_path" not in kwargs and (
                    _database_index is None or len(args) <= _database_index
                ):
                    kwargs["database_path"] = self.database_path
                return _function(*args, **kwargs)

            self.stack.enter_context(patch.object(server.attendance_db, name, bound))
        server.HOD_SESSIONS.clear()
        server.HOD_LOGIN_FAILURES.clear()
        server.HOD_LOGIN_LOCKOUTS.clear()
        server.ADMIN_SESSIONS.clear()

    def tearDown(self):
        server.HOD_SESSIONS.clear()
        server.ADMIN_SESSIONS.clear()
        self.stack.close()
        self.temp_dir.cleanup()

    def admin_request(self):
        token = "test-admin-session"
        server.ADMIN_SESSIONS[token] = time.time() + 60
        return SimpleNamespace(cookies={server.SESSION_COOKIE: token})

    def hod_request(self):
        response = Response()
        server.hod_login(
            server.HODLoginPayload(identifier="alex@example.test", password="hod-password-123"),
            SimpleNamespace(cookies={}, client=SimpleNamespace(host="127.0.0.1"), url=SimpleNamespace(scheme="http")),
            response,
        )
        cookie = SimpleCookie()
        for key, value in response.raw_headers:
            if key.lower() == b"set-cookie":
                cookie.load(value.decode())
        return SimpleNamespace(cookies={server.HOD_SESSION_COOKIE: cookie[server.HOD_SESSION_COOKIE].value})

    def test_hr_creates_hod_without_creating_employee_record(self):
        employee_count = len(database.employees(database_path=self.database_path))
        result = server.create_hod_account(
            server.HODAccountCreate(
                full_name="Mina HOD",
                email="mina@example.test",
                department="Finance",
                password="mina-password-123",
            ),
            self.admin_request(),
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(database.employees(database_path=self.database_path)), employee_count)
        accounts = server.hod_account_list(self.admin_request())
        self.assertEqual(len(accounts), 2)
        self.assertNotIn("password_hash", accounts[0])

    def test_hod_dashboard_only_contains_assigned_department(self):
        request = self.hod_request()
        dashboard = server.hod_dashboard(request)
        self.assertEqual(dashboard["department"], "Operations")
        self.assertEqual(dashboard["stats"]["total_employees"], 1)
        self.assertEqual(
            {item["employee_id"] for item in dashboard["employees"]},
            {self.alice["employee_id"]},
        )

    def test_hod_cannot_access_employee_from_another_department(self):
        request = self.hod_request()
        with self.assertRaises(HTTPException) as error:
            server.hod_employee_detail(self.bob["employee_id"], request)
        self.assertEqual(error.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
