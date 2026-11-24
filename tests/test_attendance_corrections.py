import os
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import database


class AttendanceCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "attendance.db")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff({"Alice": []}, {}, self.database_path)
        self.employee = database.employees(database_path=self.database_path)[0]
        database.update_employee(
            self.employee["employee_id"],
            {"department": "Operations", "designation": "Analyst"},
            self.database_path,
        )
        self.qatar = ZoneInfo("Asia/Qatar")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duplicate_request_and_approved_checkout_preserve_original(self):
        day = datetime(2026, 11, 20, 9, 0, tzinfo=self.qatar)
        database.record_attendance(
            self.employee["employee_id"], "check_in", day,
            database_path=self.database_path,
        )
        correction_id = database.create_attendance_correction_request(
            self.employee["employee_id"], "2026-11-20", None, "17:30",
            "Forgot to use face checkout", "Forgot to checkout", "",
            self.database_path,
        )
        self.assertIsNotNone(correction_id)
        duplicate = database.create_attendance_correction_request(
            self.employee["employee_id"], "2026-11-20", None, "18:00",
            "Second request", "Forgot to checkout", "",
            self.database_path,
        )
        self.assertIsNone(duplicate)
        reviewed = database.review_attendance_correction(
            correction_id, "approved", "Verified meeting record",
            "HOD-000001", "Operations HOD", "hod", self.database_path,
        )
        self.assertTrue(reviewed)
        correction = database.attendance_correction_requests(
            database_path=self.database_path,
        )[0]
        self.assertEqual(correction["status"], "approved")
        self.assertEqual(correction["approver_role"], "hod")
        self.assertEqual(correction["approved_check_out"][:10], "2026-11-20")
        self.assertIsNone(correction["original_check_out"])
        record = database.attendance_record_by_employee_date(
            self.employee["employee_id"], "2026-11-20", self.database_path,
        )
        self.assertEqual(record["working_hours"], 8.5)
        audit = database.correction_audit(correction_id, self.database_path)
        self.assertTrue(any(item["action"] == "submitted" for item in audit))
        self.assertTrue(any(item["action"] == "approved" for item in audit))

    def test_day_wise_summary_is_unique_and_excludes_temporary_break_time(self):
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee["employee_id"], "check_in",
            datetime(2026, 11, 20, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee["employee_id"], "check_out",
            datetime(2026, 11, 20, 12, 30, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee["employee_id"], "check_in",
            datetime(2026, 11, 20, 13, 30, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee["employee_id"], "check_out",
            datetime(2026, 11, 20, 17, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        rows = database.day_wise_attendance(
            "2026-11-20", "2026-11-20", database_path=self.database_path,
        )
        own = [row for row in rows if row["employee_id"] == self.employee["employee_id"]]
        self.assertEqual(len(own), 1)
        self.assertEqual(own[0]["break_duration"], 1.0)
        self.assertEqual(own[0]["working_hours"], 7.0)

    def test_missing_checkout_is_marked_after_shift_end_without_inventing_time(self):
        day = datetime(2026, 11, 20, 9, 0, tzinfo=self.qatar)
        record = database.record_attendance(
            self.employee["employee_id"], "check_in", day,
            database_path=self.database_path,
        )
        after_shift = datetime(2026, 11, 20, 19, 0, tzinfo=self.qatar)
        marked = database.detect_missing_checkouts(after_shift, self.database_path)
        self.assertEqual(marked, 1)
        saved = database.attendance_record(record["attendance_id"], self.database_path)
        self.assertEqual(saved["check_out"], None)
        self.assertEqual(saved["missing_checkout_pending"], 1)


if __name__ == "__main__":
    unittest.main()
