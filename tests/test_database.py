import csv
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import database
import server


class AttendanceDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "attendance.db")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff({"Alice": []}, {}, self.database_path)
        employee = database.employee_by_face_name("Alice", self.database_path)
        database.update_employee(
            employee["employee_id"], {"consent_at": "2026-01-01T08:00:00+03:00"},
            self.database_path,
        )
        self.employee_id = employee["employee_id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_in_and_check_out_calculate_hours_in_qatar_time(self):
        qatar = ZoneInfo("Asia/Qatar")
        checked_in = database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 5, tzinfo=qatar),
            database_path=self.database_path,
        )
        self.assertEqual(checked_in["attendance_date"], "2026-01-05")
        self.assertEqual(checked_in["attendance_status"], "GRACE_PERIOD")
        self.assertEqual(checked_in["late"], 0)
        checked_out = database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 5, 17, 5, tzinfo=qatar),
            database_path=self.database_path,
        )
        self.assertEqual(checked_out["status"], "checked_out")
        self.assertEqual(checked_out["working_hours"], 8.0)
        self.assertEqual(checked_out["early_departure"], 1)

    def test_check_in_and_check_out_store_separate_evidence(self):
        qatar = ZoneInfo("Asia/Qatar")
        checked_in = database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="checkin/2026/01/checkin.jpg",
            recognition_score=0.91,
            liveness_score=0.92,
        )
        self.assertEqual(checked_in["checkin_photo_path"], "checkin/2026/01/checkin.jpg")
        self.assertIsNone(checked_in["checkout_photo_path"])
        self.assertEqual(checked_in["checkin_photo_timestamp"], checked_in["check_in"])
        self.assertEqual(checked_in["checkin_recognition_score"], 0.91)
        self.assertEqual(checked_in["checkin_liveness_score"], 0.92)

        checked_out = database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 5, 18, 0, tzinfo=qatar),
            database_path=self.database_path,
            photo_path="checkout/2026/01/checkout.jpg",
            recognition_score=0.94,
            liveness_score=0.93,
        )
        self.assertEqual(
            checked_out["checkin_photo_path"], "checkin/2026/01/checkin.jpg"
        )
        self.assertEqual(
            checked_out["checkout_photo_path"], "checkout/2026/01/checkout.jpg"
        )
        self.assertEqual(checked_out["checkout_photo_timestamp"], checked_out["check_out"])
        self.assertEqual(checked_out["checkout_recognition_score"], 0.94)
        self.assertEqual(checked_out["checkout_liveness_score"], 0.93)
        self.assertEqual(
            database.attendance_photo_path(
                checked_out["attendance_id"], "check_in", self.database_path
            ),
            "checkin/2026/01/checkin.jpg",
        )
        self.assertEqual(
            database.clear_attendance_photo(
                checked_out["attendance_id"], "check_in", self.database_path
            ),
            "checkin/2026/01/checkin.jpg",
        )
        cleared = database.attendance_record(
            checked_out["attendance_id"], self.database_path
        )
        self.assertIsNone(cleared["checkin_photo_path"])
        self.assertIsNone(cleared["checkin_liveness_score"])
        self.assertEqual(
            cleared["checkout_photo_path"], "checkout/2026/01/checkout.jpg"
        )

    def test_duplicate_check_in_and_check_out_without_check_in_are_rejected(self):
        qatar = ZoneInfo("Asia/Qatar")
        now = datetime(2026, 1, 5, 9, 0, tzinfo=qatar)
        with self.assertRaises(LookupError):
            database.record_attendance(
                self.employee_id, "check_out", now, database_path=self.database_path
            )
        database.record_attendance(
            self.employee_id, "check_in", now, database_path=self.database_path
        )
        with self.assertRaises(FileExistsError):
            database.record_attendance(
                self.employee_id, "check_in", now, database_path=self.database_path
            )

    def test_return_after_temporary_exits_preserves_first_in_and_final_out(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id, "09:00", "17:00", 30, self.database_path,
            on_time_window_minutes=15,
        )
        qatar = ZoneInfo("Asia/Qatar")

        def clock(action, hour, minute, photo):
            return database.record_attendance(
                self.employee_id,
                action,
                datetime(2026, 1, 5, hour, minute, tzinfo=qatar),
                database_path=self.database_path,
                photo_path=photo,
                liveness_score=0.91,
            )

        initial = clock("check_in", 9, 0, "checkin.jpg")
        early_exit = clock("check_out", 12, 30, "lunch-exit.jpg")
        self.assertTrue(early_exit["early_departure"])
        self.assertEqual(early_exit["early_departure_minutes"], 270)
        returned = clock("check_in", 13, 30, "lunch-return.jpg")
        self.assertTrue(returned["returned_from_break"])
        self.assertEqual(returned["attendance_id"], initial["attendance_id"])
        self.assertEqual(returned["check_in"], initial["check_in"])
        self.assertIsNone(returned["check_out"])
        self.assertIsNone(returned["working_hours"])
        self.assertEqual(returned["checkin_photo_path"], "checkin.jpg")
        self.assertIsNone(returned["checkout_photo_path"])

        final_exit = clock("check_out", 17, 0, "final-exit.jpg")
        self.assertEqual(final_exit["check_out"], "2026-01-05T17:00:00+03:00")
        self.assertEqual(final_exit["working_hours"], 8.0)
        self.assertFalse(final_exit["early_departure"])
        movements = database.attendance_movements(initial["attendance_id"], self.database_path)
        self.assertEqual(
            [(item["action"], item["occurred_at"]) for item in movements],
            [
                ("check_out", "2026-01-05T12:30:00+03:00"),
                ("check_in", "2026-01-05T13:30:00+03:00"),
            ],
        )
        self.assertEqual(
            {item["photo_path"] for item in movements},
            {"lunch-exit.jpg", "lunch-return.jpg"},
        )
        self.assertEqual(
            {item["liveness_score"] for item in movements},
            {0.91},
        )
        self.assertEqual(final_exit["checkout_liveness_score"], 0.91)
        self.assertEqual(
            set(database.attendance_photo_paths(self.database_path)),
            {"checkin.jpg", "lunch-exit.jpg", "lunch-return.jpg", "final-exit.jpg"},
        )
        self.assertEqual(
            len(database.attendance_records(database_path=self.database_path)), 1
        )
        database.clear_attendance_records(self.database_path)
        self.assertEqual(database.attendance_photo_paths(self.database_path), [])
        self.assertEqual(
            database.attendance_movements(initial["attendance_id"], self.database_path),
            [],
        )

    def test_early_departure_without_return_keeps_final_exit_and_minutes(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id, "09:00", "17:00", 30, self.database_path,
            on_time_window_minutes=15,
        )
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id, "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        departure = database.record_attendance(
            self.employee_id, "check_out",
            datetime(2026, 1, 5, 12, 30, tzinfo=qatar),
            database_path=self.database_path,
        )
        self.assertEqual(departure["check_in"], "2026-01-05T09:00:00+03:00")
        self.assertEqual(departure["check_out"], "2026-01-05T12:30:00+03:00")
        self.assertEqual(departure["working_hours"], 3.5)
        self.assertTrue(departure["early_departure"])
        self.assertEqual(departure["early_departure_minutes"], 270)
        report = database.analytics(
            "2026-01-05", "2026-01-05", self.database_path
        )
        self.assertEqual(report["daily"][0]["early_departures"], 1)
        self.assertEqual(report["daily"][0]["average_hours"], 3.5)
        self.assertEqual(report["departments"][0]["early_departures"], 1)
        with self.assertRaises(ValueError):
            database.record_attendance(
                self.employee_id, "check_in",
                datetime(2026, 1, 5, 12, 0, tzinfo=qatar),
                database_path=self.database_path,
            )

    def test_legacy_attendance_events_remain_separate_history_rows(self):
        events = (
            ("2026-01-05T09:00:00+03:00", "checked_in"),
            ("2026-01-05T12:00:00+03:00", "checked_out"),
            ("2026-01-05T13:00:00+03:00", "checked_in"),
        )
        for timestamp, status in events:
            database.log_access_event(
                {
                    "time": timestamp,
                    "event": "attendance",
                    "status": status,
                    "name": "Alice",
                    "score": "",
                    "method": "face",
                    "details": "",
                },
                self.database_path,
            )

        rows = database.legacy_attendance_records(
            "2026-01-05", "2026-01-05", database_path=self.database_path
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["check_in"] for row in rows],
            ["2026-01-05T09:00:00+03:00", "2026-01-05T13:00:00+03:00"],
        )
        self.assertEqual(rows[0]["check_out"], "2026-01-05T12:00:00+03:00")
        self.assertEqual(rows[0]["status"], "checked_out")
        self.assertIsNone(rows[1]["check_out"])
        self.assertTrue(all(row["legacy_event"] for row in rows))
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )

    def test_overnight_shift_keeps_schedule_date_and_blocks_another_open_shift(self):
        shift_id = database.create_shift(
            "Night", "22:00", "06:00", 15, 8, self.database_path
        )
        database.set_employee_shift(self.employee_id, shift_id, self.database_path)
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 22, 15, tzinfo=qatar),
            database_path=self.database_path,
        )
        with self.assertRaises(FileExistsError):
            database.record_attendance(
                self.employee_id,
                "check_in",
                datetime(2026, 1, 6, 2, 0, tzinfo=qatar),
                database_path=self.database_path,
            )
        checked_out = database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 6, 6, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        self.assertEqual(checked_out["attendance_date"], "2026-01-05")
        self.assertEqual(checked_out["working_hours"], 7.75)
        self.assertEqual(checked_out["late"], 0)
        self.assertEqual(checked_out["early_departure"], 0)

    def test_remove_assigned_shift_requires_manual_reassignment(self):
        default_shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        extra_shift_id = database.create_shift(
            "Early", "09:00", "18:00", 15, 8, self.database_path
        )
        with self.assertRaisesRegex(ValueError, "Assign every employee"):
            database.remove_shift(default_shift_id, self.database_path)
        database.set_employee_shift(self.employee_id, extra_shift_id, self.database_path)
        result = database.remove_shift(default_shift_id, self.database_path)
        self.assertFalse(result["attendance_preserved"])
        self.assertEqual(
            database.employee_by_id(self.employee_id, self.database_path)["shift_id"],
            extra_shift_id,
        )
        self.assertNotIn(
            default_shift_id,
            {shift["shift_id"] for shift in database.list_shifts(self.database_path)},
        )

    def test_update_shift_times_preserves_assignments_and_attendance_history(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 5, 18, 0, tzinfo=qatar),
            database_path=self.database_path,
        )

        self.assertTrue(
            database.update_shift_times(
                shift_id, "08:00", "17:00", self.database_path
            )
        )
        new_schedule_attendance = database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 6, 8, 10, tzinfo=qatar),
            database_path=self.database_path,
        )
        self.assertEqual(new_schedule_attendance["late"], 0)
        shift = next(
            row for row in database.list_shifts(self.database_path)
            if row["shift_id"] == shift_id
        )
        self.assertEqual((shift["start_time"], shift["end_time"]), ("08:00", "17:00"))
        self.assertEqual(
            database.employee_by_id(self.employee_id, self.database_path)["shift_id"],
            shift_id,
        )
        self.assertEqual(
            database.attendance_records(database_path=self.database_path)[0]["shift_name"],
            "Default",
        )
        self.assertFalse(
            database.update_shift_times(
                shift_id + 100, "08:00", "17:00", self.database_path
            )
        )

    def test_remove_shift_preserves_attendance_history(self):
        shift_id = database.create_shift(
            "Early", "09:00", "18:00", 15, 8, self.database_path
        )
        database.set_employee_shift(self.employee_id, shift_id, self.database_path)
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.set_employee_shift(
            self.employee_id,
            database.create_shift(
                "Replacement", "09:00", "18:00", 15, 8, self.database_path
            ),
            self.database_path,
        )
        result = database.remove_shift(shift_id, self.database_path)
        self.assertTrue(result["attendance_preserved"])
        self.assertNotIn(
            shift_id,
            {shift["shift_id"] for shift in database.list_shifts(self.database_path)},
        )
        attendance = database.attendance_records(database_path=self.database_path)
        self.assertEqual(attendance[0]["shift_name"], "Early")
        with database.connection(self.database_path) as db:
            archived = db.execute(
                "SELECT removed_at, active FROM shifts WHERE shift_id = ?",
                (shift_id,),
            ).fetchone()
        self.assertIsNotNone(archived["removed_at"])
        self.assertEqual(archived["active"], 0)

    def test_remove_unassigned_shift_without_history_and_report_missing_shift(self):
        shift_id = database.create_shift(
            "Temporary", "10:00", "19:00", 10, 8, self.database_path
        )
        self.assertTrue(database.remove_shift(shift_id, self.database_path))
        self.assertFalse(database.remove_shift(shift_id, self.database_path))

    def test_remove_default_shift_when_unused_and_do_not_reseed_it(self):
        default_shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        alternate_id = database.create_shift(
            "Alternate", "09:00", "18:00", 15, 8, self.database_path
        )
        database.set_employee_shift(self.employee_id, alternate_id, self.database_path)

        self.assertTrue(database.remove_shift(default_shift_id, self.database_path))
        database.initialize_database(self.database_path)
        self.assertNotIn(
            default_shift_id,
            {shift["shift_id"] for shift in database.list_shifts(self.database_path)},
        )

    def test_clear_attendance_history_keeps_employees_and_access_events(self):
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.log_access_event({
            "time": "2026-01-05T09:00:00+03:00",
            "event": "attendance",
            "status": "checked_in",
            "name": "Alice",
            "score": "0.9",
            "method": "face",
            "details": "",
        }, self.database_path)

        self.assertEqual(
            database.clear_attendance_records(self.database_path), 1
        )
        self.assertEqual(
            database.attendance_records(database_path=self.database_path), []
        )
        self.assertIsNotNone(
            database.employee_by_id(self.employee_id, self.database_path)
        )
        with database.connection(self.database_path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM access_events").fetchone()[0], 1)
        self.assertEqual(
            database.legacy_attendance_records(database_path=self.database_path), []
        )
        database.log_access_event({
            "time": "2026-01-06T09:00:00+03:00",
            "event": "attendance",
            "status": "checked_in",
            "name": "Alice",
            "score": "0.9",
            "method": "face",
            "details": "",
        }, self.database_path)
        self.assertEqual(
            len(database.legacy_attendance_records(database_path=self.database_path)),
            1,
        )
        self.assertEqual(
            database.clear_attendance_records(self.database_path), 0
        )
        self.assertEqual(
            database.legacy_attendance_records(database_path=self.database_path), []
        )

    def test_cannot_disable_or_remove_last_active_shift_assigned_to_employee(self):
        isolated_path = os.path.join(self.temp_dir.name, "isolated.db")
        database.initialize_database(isolated_path)
        default_shift_id = database.list_shifts(isolated_path)[0]["shift_id"]
        with self.assertRaisesRegex(ValueError, "At least one active shift"):
            database.set_shift_active(default_shift_id, False, isolated_path)
        with self.assertRaisesRegex(ValueError, "replacement shift"):
            database.remove_shift(default_shift_id, isolated_path)

    def test_can_disable_default_shift_when_another_active_shift_exists(self):
        default_shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.create_shift(
            "Alternate", "09:00", "18:00", 15, 8, self.database_path
        )
        self.assertTrue(
            database.set_shift_active(default_shift_id, False, self.database_path)
        )
        default = next(
            shift for shift in database.list_shifts(self.database_path)
            if shift["shift_id"] == default_shift_id
        )
        self.assertEqual(default["active"], 0)

    def test_enable_shift_restores_previously_disabled_shift(self):
        shift_id = database.create_shift(
            "Legacy disabled", "10:00", "19:00", 10, 8, self.database_path
        )
        with database.connection(self.database_path) as db:
            db.execute("UPDATE shifts SET active = 0 WHERE shift_id = ?", (shift_id,))
        self.assertTrue(database.enable_shift(shift_id, self.database_path))
        shift = next(
            item for item in database.list_shifts(self.database_path)
            if item["shift_id"] == shift_id
        )
        self.assertEqual(shift["active"], 1)
        self.assertFalse(database.enable_shift(999, self.database_path))

    def test_check_in_after_grace_is_recorded_and_marked_separately(self):
        qatar = ZoneInfo("Asia/Qatar")
        for day, hour, minute, second, expected_late, expected_after_grace, expected_status in (
            (5, 8, 59, 59, 0, 0, "ON_TIME"),
            (6, 9, 0, 0, 0, 0, "ON_TIME"),
            (7, 9, 15, 0, 0, 0, "GRACE_PERIOD"),
            (8, 9, 15, 1, 1, 1, "LATE"),
        ):
            checked_in = database.record_attendance(
                self.employee_id,
                "check_in",
                datetime(2026, 1, day, hour, minute, second, tzinfo=qatar),
                database_path=self.database_path,
            )
            self.assertEqual(checked_in["late"], expected_late)
            self.assertEqual(checked_in["late_after_grace"], expected_after_grace)
            self.assertEqual(checked_in["attendance_status"], expected_status)
            self.assertEqual(checked_in["status"], "present")
            database.record_attendance(
                self.employee_id,
                "check_out",
                datetime(2026, 1, day, 18, 0, tzinfo=qatar),
                database_path=self.database_path,
            )

        self.assertEqual(
            len(database.attendance_records(database_path=self.database_path)), 4
        )

    def test_punctuality_categories_and_lateness_minutes_use_current_shift_grace(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id, "09:00", "18:00", 10, self.database_path
        )
        cases = (
            (1, 8, 45, "ON_TIME", 0, 0, 0),
            (2, 9, 0, "ON_TIME", 0, 0, 0),
            (3, 9, 5, "GRACE_PERIOD", 5, 5, 0),
            (4, 9, 10, "GRACE_PERIOD", 10, 10, 0),
            (5, 9, 11, "LATE", 11, 10, 1),
            (6, 9, 30, "LATE", 30, 10, 20),
            (7, 9, 10, "LATE", 10, 10, 1),
        )
        qatar = ZoneInfo("Asia/Qatar")
        for day, hour, minute, expected, after_start, grace_used, late_by in cases:
            row = database.record_attendance(
                self.employee_id,
                "check_in",
                datetime(
                    2026, 1, day, hour, minute,
                    second=1 if day == 7 else 0,
                    tzinfo=qatar,
                ),
                database_path=self.database_path,
            )
            self.assertEqual(row["attendance_status"], expected)
            self.assertEqual(row["minutes_after_start"], after_start)
            self.assertEqual(row["grace_minutes_used"], grace_used)
            self.assertEqual(row["late_minutes"], late_by)
            database.record_attendance(
                self.employee_id, "check_out",
                datetime(2026, 1, day, 18, 30, tzinfo=qatar),
                database_path=self.database_path,
            )

        database.update_shift_configuration(
            shift_id, "09:15", "18:00", 5, self.database_path
        )
        recalculated = database.attendance_records(
            "2026-01-05", "2026-01-05", database_path=self.database_path
        )[0]
        self.assertEqual(recalculated["attendance_status"], "ON_TIME")
        self.assertEqual(recalculated["check_in"], "2026-01-05T09:11:00+03:00")

    def test_on_time_grace_and_late_boundaries_for_9_to_5_shift(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id,
            "09:00",
            "17:00",
            30,
            self.database_path,
            on_time_window_minutes=15,
        )
        cases = (
            (1, 8, 59, 59, "ON_TIME", 0),
            (2, 9, 0, 0, "ON_TIME", 0),
            (3, 9, 14, 59, "ON_TIME", 0),
            (4, 9, 15, 0, "GRACE_PERIOD", 0),
            (5, 9, 30, 0, "GRACE_PERIOD", 0),
            (6, 9, 30, 1, "LATE", 1),
            (7, 9, 31, 0, "LATE", 1),
        )
        qatar = ZoneInfo("Asia/Qatar")
        for day, hour, minute, second, expected_status, late_minutes in cases:
            row = database.record_attendance(
                self.employee_id,
                "check_in",
                datetime(
                    2026, 1, day, hour, minute, second, tzinfo=qatar
                ),
                database_path=self.database_path,
            )
            self.assertEqual(row["attendance_status"], expected_status)
            self.assertEqual(row["late_minutes"], late_minutes)
            database.record_attendance(
                self.employee_id,
                "check_out",
                datetime(2026, 1, day, 17, 0, tzinfo=qatar),
                database_path=self.database_path,
            )
        daily = database.daily_attendance(
            "2026-01-04",
            datetime(2026, 1, 4, 18, 0, tzinfo=qatar),
            self.database_path,
        )
        self.assertEqual(daily["counts"]["CHECKED_OUT"], 1)
        self.assertEqual(daily["punctuality_counts"]["GRACE_PERIOD"], 1)

    def test_on_time_window_cannot_exceed_late_cutoff(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            database.update_shift_configuration(
                shift_id,
                "09:00",
                "17:00",
                15,
                self.database_path,
                on_time_window_minutes=16,
            )

    def test_hr_late_filter_includes_employee_who_already_checked_out(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id,
            "09:00",
            "17:00",
            30,
            self.database_path,
            on_time_window_minutes=15,
        )
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 9, 31, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 5, 17, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        daily_attendance = database.daily_attendance
        legacy_attendance_records = database.legacy_attendance_records
        with (
            patch.object(
                server.attendance_db,
                "daily_attendance",
                side_effect=lambda day: daily_attendance(
                    day, datetime(2026, 1, 5, 18, 0, tzinfo=qatar),
                    self.database_path,
                ),
            ),
            patch.object(
                server.attendance_db,
                "legacy_attendance_records",
                side_effect=lambda start="", end="", name="": legacy_attendance_records(
                    start, end, name, self.database_path
                ),
            ),
        ):
            _, filtered, _, _, punctuality_counts, _, _ = (
                server.attendance_monitor_data(
                    "2026-01-05", "2026-01-05", datetime(2026, 1, 5).date(),
                    status="LATE",
                )
            )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["status"], "CHECKED_OUT")
        self.assertEqual(filtered[0]["attendance_status"], "LATE")
        self.assertEqual(punctuality_counts["LATE"], 1)

    def test_same_day_recheckin_uses_one_attendance_record_in_monitor(self):
        qatar = ZoneInfo("Asia/Qatar")
        first_checkin = datetime(2026, 1, 5, 9, 0, tzinfo=qatar)
        database.record_attendance(
            self.employee_id,
            "check_in",
            first_checkin,
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employee_id,
            "check_out",
            datetime(2026, 1, 5, 12, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        returned = database.record_attendance(
            self.employee_id,
            "check_in",
            datetime(2026, 1, 5, 13, 0, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.log_access_event(
            {
                "time": "2026-01-05T13:01:00+03:00",
                "event": "attendance",
                "status": "checked_in",
                "name": "Alice",
                "score": "",
                "method": "face",
                "details": "",
            },
            self.database_path,
        )
        daily_attendance = database.daily_attendance
        legacy_attendance_records = database.legacy_attendance_records
        with (
            patch.object(
                server.attendance_db,
                "daily_attendance",
                side_effect=lambda day: daily_attendance(
                    day, datetime(2026, 1, 5, 14, 0, tzinfo=qatar),
                    self.database_path,
                ),
            ),
            patch.object(
                server.attendance_db,
                "legacy_attendance_records",
                side_effect=lambda start="", end="", name="": legacy_attendance_records(
                    start, end, name, self.database_path
                ),
            ),
        ):
            all_records, records, *_ = server.attendance_monitor_data(
                "2026-01-05",
                "2026-01-05",
                first_checkin.date(),
            )

        self.assertEqual(len(all_records), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["attendance_id"], returned["attendance_id"])
        self.assertEqual(records[0]["check_in"], first_checkin.isoformat())
        self.assertEqual(records[0]["status"], "CURRENTLY_WORKING")
        self.assertNotIn("legacy_event", records[0])

    def test_daily_statuses_respect_deadline_leave_workdays_and_holidays(self):
        database.sync_legacy_staff(
            {"Alice": [], "Bob": []}, {}, self.database_path
        )
        bob = database.employee_by_face_name("Bob", self.database_path)
        leave_id = database.create_leave_request(
            bob["employee_id"], "2026-01-07", "2026-01-07",
            "Approved leave", self.database_path,
        )
        self.assertTrue(
            database.update_leave_status(leave_id, "approved", self.database_path)
        )
        database.update_settings(
            {
                "working_days": '["sun","mon","tue","wed","thu"]',
                "holidays": '["2026-01-08"]',
            },
            self.database_path,
        )
        qatar = ZoneInfo("Asia/Qatar")

        before_deadline = database.daily_attendance(
            "2026-01-07",
            datetime(2026, 1, 7, 9, 15, tzinfo=qatar),
            self.database_path,
        )
        self.assertEqual(before_deadline["counts"]["NOT_CHECKED_IN"], 1)
        self.assertEqual(before_deadline["counts"]["ON_LEAVE"], 1)

        after_deadline = database.daily_attendance(
            "2026-01-07",
            datetime(2026, 1, 7, 9, 16, tzinfo=qatar),
            self.database_path,
        )
        self.assertEqual(after_deadline["counts"]["ABSENT"], 1)
        self.assertEqual(after_deadline["counts"]["ON_LEAVE"], 1)
        self.assertEqual(after_deadline["total_scheduled"], 1)

        holiday = database.daily_attendance(
            "2026-01-08", datetime(2026, 1, 8, 12, tzinfo=qatar),
            self.database_path,
        )
        self.assertEqual(holiday["counts"]["HOLIDAY"], 2)

        weekly_off = database.daily_attendance(
            "2026-01-09", datetime(2026, 1, 9, 12, tzinfo=qatar),
            self.database_path,
        )
        self.assertEqual(weekly_off["counts"]["WEEKLY_OFF"], 2)
        dashboard = database.dashboard_stats("2026-01-09", self.database_path)
        self.assertEqual(dashboard["total_employees"], 0)
        self.assertEqual(dashboard["absent_today"], 0)

    def test_default_grace_period_is_15_minutes_and_migrates_old_default(self):
        self.assertEqual(database.settings(self.database_path)["grace_period_minutes"], "15")
        with database.connection(self.database_path) as db:
            db.execute(
                "UPDATE shifts SET grace_period_minutes = 10 WHERE shift_id = 1"
            )
            db.execute(
                """UPDATE app_settings SET setting_value = '10'
                   WHERE setting_key = 'grace_period_minutes'"""
            )
            db.execute("DELETE FROM schema_migrations WHERE version = 2")

        database.initialize_database(self.database_path)
        self.assertEqual(database.settings(self.database_path)["grace_period_minutes"], "15")
        default = next(
            shift for shift in database.list_shifts(self.database_path)
            if shift["shift_id"] == 1
        )
        self.assertEqual(default["grace_period_minutes"], 15)

    def test_legacy_access_rows_are_imported_once_without_becoming_attendance(self):
        csv_path = os.path.join(self.temp_dir.name, "entries.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=database_access_fields())
            writer.writeheader()
            writer.writerow({
                "time": "2026-01-05T09:00:00",
                "event": "entry",
                "status": "granted",
                "name": "Alice",
                "score": "0.91",
                "method": "face",
                "details": "",
            })
        self.assertEqual(database.import_legacy_entries(csv_path, self.database_path), 1)
        self.assertEqual(database.import_legacy_entries(csv_path, self.database_path), 0)
        with database.connection(self.database_path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM access_events").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 0)

    def test_existing_database_access_event_is_not_reimported_from_csv(self):
        csv_path = os.path.join(self.temp_dir.name, "entries.csv")
        entry = {
            "time": "2026-01-05T09:00:00",
            "event": "entry",
            "status": "granted",
            "name": "Alice",
            "score": "0.91",
            "method": "face",
            "details": "",
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=database_access_fields())
            writer.writeheader()
            writer.writerow(entry)
        database.log_access_event(entry, self.database_path)
        self.assertEqual(database.import_legacy_entries(csv_path, self.database_path), 0)
        with database.connection(self.database_path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM access_events").fetchone()[0], 1)

    def test_reenrollment_preserves_status_and_assigned_shift(self):
        shift_id = database.create_shift(
            "Early", "07:00", "16:00", 5, 8, self.database_path
        )
        database.set_employee_shift(self.employee_id, shift_id, self.database_path)
        database.update_employee(
            self.employee_id,
            {"status": "inactive", "department": "Operations"},
            self.database_path,
        )
        database.upsert_enrolled_employee(
            "Alice", {"profile_photo": "new-photo.jpg"}, self.database_path
        )
        employee = database.employee_by_id(self.employee_id, self.database_path)
        self.assertEqual(employee["status"], "inactive")
        self.assertEqual(employee["shift_id"], shift_id)
        self.assertEqual(employee["department"], "Operations")


def database_access_fields():
    return ["time", "event", "status", "name", "score", "method", "details"]


if __name__ == "__main__":
    unittest.main()
