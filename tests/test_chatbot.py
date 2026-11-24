import os
import tempfile
import unittest
import zipfile
from datetime import datetime
from io import BytesIO
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import chatbot_service
import database


class AttendanceChatbotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "chatbot.db")
        database.initialize_database(self.database_path)
        database.sync_legacy_staff(
            {"Alice": [], "Bob": [], "Carol": []}, {}, self.database_path
        )
        self.employees = {
            employee["face_name"]: employee
            for employee in database.employees(database_path=self.database_path)
        }
        for name, department in (("Alice", "IT"), ("Bob", "IT"), ("Carol", "HR")):
            database.update_employee(
                self.employees[name]["employee_id"],
                {"department": department},
                self.database_path,
            )
        self.now = datetime(2026, 11, 18, 10, 0, tzinfo=ZoneInfo("Asia/Qatar"))
        database.record_attendance(
            self.employees["Alice"]["employee_id"],
            "check_in",
            datetime(2026, 11, 18, 9, 5, tzinfo=ZoneInfo("Asia/Qatar")),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employees["Bob"]["employee_id"],
            "check_in",
            datetime(2026, 11, 18, 8, 45, tzinfo=ZoneInfo("Asia/Qatar")),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employees["Bob"]["employee_id"],
            "check_out",
            datetime(2026, 11, 18, 17, 0, tzinfo=ZoneInfo("Asia/Qatar")),
            database_path=self.database_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def ask(self, question, context=None):
        return chatbot_service.answer_question(
            question,
            context,
            database_path=self.database_path,
            now=self.now,
        )

    def test_daily_absent_late_on_time_and_working_statuses(self):
        absent = self.ask("Who is absent today?")["answer"]
        self.assertIn("Carol", absent)
        self.assertNotIn("Alice", absent)
        self.assertIn("after each assigned shift", absent)

        late = self.ask("Who came late today?")["answer"]
        self.assertIn("None.", late)
        self.assertIn("after the configured grace period", late)

        on_time = self.ask("Who arrived on time today?")["answer"]
        self.assertIn("Bob", on_time)
        self.assertNotIn("Alice", on_time)
        punctuality = self.ask("What percentage arrived on time today?")["answer"]
        self.assertIn("50.0%", punctuality)

        working = self.ask("Who is currently working?")["answer"]
        self.assertIn("Alice", working)
        self.assertNotIn("Bob", working)

    def test_employee_checkin_checkout_and_unknown_employee(self):
        checkin = self.ask("What time did Alice check in today?")["answer"]
        self.assertIn("09:05 AM", checkin)
        self.assertIn("Working hours are not final", checkin)

        employee_status = self.ask("Is Alice present today?")
        self.assertIn("GRACE_PERIOD - CURRENTLY WORKING", employee_status["answer"])
        self.assertEqual(employee_status["status"], "GRACE_PERIOD - CURRENTLY WORKING")

        checkout = self.ask("What time did Bob check out today?")["answer"]
        self.assertIn("05:00 PM", checkout)
        self.assertIn("Completed hours: 8.25", checkout)

        unknown = self.ask("Is Nobody present today?")["answer"]
        self.assertIn("couldn't find", unknown)

    def test_present_and_checked_out_questions_return_matching_evidence_photos(self):
        records = database.attendance_records(
            "2026-11-18", "2026-11-18", database_path=self.database_path
        )
        with database.connection(self.database_path) as db:
            for record in records:
                if record["full_name"] == "Alice":
                    db.execute(
                        """UPDATE attendance
                           SET checkin_photo_path = ?, checkin_photo_timestamp = ?
                           WHERE attendance_id = ?""",
                        ("alice-checkin.jpg", "2026-11-18T09:05:00+03:00",
                         record["attendance_id"]),
                    )
                elif record["full_name"] == "Bob":
                    db.execute(
                        """UPDATE attendance
                           SET checkin_photo_path = ?, checkout_photo_path = ?,
                               checkin_photo_timestamp = ?, checkout_photo_timestamp = ?
                           WHERE attendance_id = ?""",
                        ("bob-checkin.jpg", "bob-checkout.jpg",
                         "2026-11-18T08:45:00+03:00",
                         "2026-11-18T17:00:00+03:00", record["attendance_id"]),
                    )

        present = self.ask("Who all are present ryt?")
        self.assertIn("Alice", present["answer"])
        self.assertIn("Bob", present["answer"])
        self.assertEqual(
            {photo["employee_name"] for photo in present["photos"]},
            {"Alice", "Bob"},
        )
        self.assertTrue(all(photo["action"] == "check_in" for photo in present["photos"]))

        checked_out = self.ask("Who all are checkout?")
        self.assertIn("Bob", checked_out["answer"])
        self.assertNotIn("Alice", checked_out["answer"])
        self.assertIn("check-out 05:00 PM", checked_out["answer"])
        self.assertEqual(checked_out["photos"][0]["employee_name"], "Bob")
        self.assertEqual(checked_out["photos"][0]["action"], "check_out")
        self.assertEqual(
            checked_out["photos"][0]["url"],
            f"/attendance/record/{records[0]['attendance_id']}/photo/check_out"
            if records[0]["full_name"] == "Bob"
            else f"/attendance/record/{records[1]['attendance_id']}/photo/check_out",
        )

        prior = self.ask("Who came late last month?")
        current_present = self.ask(
            "Is anyone present?", prior["context"]
        )
        self.assertEqual(current_present["context"]["start_date"], "2026-11-18")
        self.assertIn("Alice", current_present["answer"])

    def test_chatbot_reports_early_departure_and_elapsed_hours_including_breaks(self):
        shift_id = database.list_shifts(self.database_path)[0]["shift_id"]
        database.update_shift_configuration(
            shift_id, "09:00", "17:00", 30, self.database_path,
            on_time_window_minutes=15,
        )
        carol_id = self.employees["Carol"]["employee_id"]
        database.record_attendance(
            carol_id,
            "check_in",
            datetime(2026, 11, 18, 9, 0, tzinfo=ZoneInfo("Asia/Qatar")),
            database_path=self.database_path,
        )
        database.record_attendance(
            carol_id,
            "check_out",
            datetime(2026, 11, 18, 12, 30, tzinfo=ZoneInfo("Asia/Qatar")),
            database_path=self.database_path,
        )
        status = self.ask("What time did Carol check out today?")["answer"]
        self.assertIn("EARLY DEPARTURE", status)
        self.assertIn("270 minute(s)", status)
        hours = self.ask("How many hours did Carol work today?")["answer"]
        self.assertIn("Completed hours: 3.50", hours)
        self.assertIn("including breaks", hours)
        self.assertIn("EARLY DEPARTURE", hours)
        summary = self.ask("Give me an attendance summary today.")["answer"]
        self.assertIn("early departure(s)", summary)

    def test_department_summary_and_monthly_summary_use_real_records(self):
        department = self.ask("How many employees are present in IT today?")["answer"]
        self.assertIn("IT", department)
        self.assertIn("2 employee(s)", department)

        monthly = self.ask("Show the monthly attendance summary.")["answer"]
        self.assertIn("Attendance records: 2", monthly)
        self.assertIn("Average completed hours: 8.25", monthly)
        overtime = self.ask("Who worked overtime today?")["answer"]
        self.assertIn("Bob: 8.25 hours", overtime)

    def test_followup_reuses_intent_and_updates_date_range(self):
        first = self.ask("Who was absent today?")
        followup = self.ask("What about last week?", first["context"])
        self.assertIn("can't reliably list historical absences", followup["answer"])
        self.assertEqual(followup["context"]["intent"], "absent")
        self.assertEqual(followup["context"]["start_date"], "2026-11-09")
        self.assertEqual(followup["context"]["end_date"], "2026-11-15")

    def test_absence_waits_until_shift_grace_deadline_and_respects_approved_leave(self):
        before_deadline = chatbot_service.answer_question(
            "Who is absent today?",
            database_path=self.database_path,
            now=datetime(2026, 11, 18, 9, 10, tzinfo=ZoneInfo("Asia/Qatar")),
        )["answer"]
        self.assertNotIn("Carol", before_deadline)

        leave_id = database.create_leave_request(
            self.employees["Carol"]["employee_id"],
            "2026-11-18",
            "2026-11-18",
            "Approved annual leave",
            self.database_path,
        )
        database.update_leave_status(leave_id, "approved", self.database_path)
        on_leave = self.ask("Who is on leave today?")["answer"]
        self.assertIn("Carol", on_leave)

    def test_employee_attendance_followup_uses_prior_date_range(self):
        first = self.ask("Show Bob's attendance for the last 7 days.")
        self.assertIn("2026-11-18", first["answer"])
        followup = self.ask("What about Alice?", first["context"])
        self.assertIn("Attendance for Alice", followup["answer"])
        self.assertIn("2026-11-18", followup["answer"])

    def test_followup_employee_name_filters_prior_intent(self):
        first = self.ask("Who came late today?")
        followup = self.ask("What about Bob?", first["context"])
        self.assertIn("None.", followup["answer"])
        self.assertNotIn("Alice", followup["answer"])

    def test_reports_generate_csv_xlsx_and_pdf(self):
        report = {
            "kind": "attendance",
            "start_date": "2026-11-18",
            "end_date": "2026-11-18",
            "department": "",
        }
        csv_data, csv_type, csv_name = chatbot_service.build_report(
            report, "csv", self.database_path
        )
        self.assertIn(b"Employee ID", csv_data)
        self.assertEqual(csv_type, "text/csv; charset=utf-8")
        self.assertTrue(csv_name.endswith(".csv"))

        xlsx_data, xlsx_type, xlsx_name = chatbot_service.build_report(
            report, "xlsx", self.database_path
        )
        workbook = BytesIO(xlsx_data)
        self.assertTrue(zipfile.is_zipfile(workbook))
        with zipfile.ZipFile(workbook) as archive:
            ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        self.assertIn("spreadsheetml", xlsx_type)
        self.assertTrue(xlsx_name.endswith(".xlsx"))

        pdf_data, pdf_type, pdf_name = chatbot_service.build_report(
            report, "pdf", self.database_path
        )
        self.assertTrue(pdf_data.startswith(b"%PDF-1.4"))
        self.assertEqual(pdf_type, "application/pdf")
        self.assertTrue(pdf_name.endswith(".pdf"))

    def test_natural_language_report_returns_live_report_parameters(self):
        result = self.ask("Generate today's attendance report.")
        self.assertEqual(result["report"]["kind"], "attendance")
        self.assertEqual(result["report"]["start_date"], "2026-11-18")
        self.assertIn("is ready", result["answer"])

    def test_compare_this_week_with_last_week_uses_both_date_ranges(self):
        answer = self.ask("Compare this week with last week.")["answer"]
        self.assertIn("2026-11-16–2026-11-18", answer)
        self.assertIn("2026-11-09–2026-11-15", answer)
        self.assertIn("2 record(s)", answer)

    def test_monthly_late_ranking_counts_each_employee_and_after_grace_arrivals(self):
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employees["Alice"]["employee_id"],
            "check_out",
            datetime(2026, 11, 18, 10, 1, tzinfo=qatar),
            database_path=self.database_path,
        )
        for day in (16, 17):
            database.record_attendance(
                self.employees["Alice"]["employee_id"],
                "check_in",
                datetime(2026, 11, day, 9, 16, tzinfo=qatar),
                database_path=self.database_path,
            )
            database.record_attendance(
                self.employees["Alice"]["employee_id"],
                "check_out",
                datetime(2026, 11, day, 18, 0, tzinfo=qatar),
                database_path=self.database_path,
            )

        answer = self.ask(
            "Which five employees have the most late arrivals this month?"
        )["answer"]
        self.assertIn("Alice (IT): 2 late arrival(s), 2 after grace", answer)
        self.assertIn("2026-11-01 through 2026-11-18", answer)

    def test_department_absenteeism_ranks_only_employees_due_for_attendance(self):
        answer = self.ask(
            "Which department has the highest absenteeism today?"
        )["answer"]
        self.assertIn("Highest absenteeism today: HR (1 of 1 scheduled, 100.0%)", answer)
        self.assertIn("IT: 0 absent of 2 scheduled (0.0%)", answer)
        self.assertIn("holiday calendars are not configured", answer)

    def test_attendance_trend_compares_completed_month_days(self):
        qatar = ZoneInfo("Asia/Qatar")
        database.record_attendance(
            self.employees["Alice"]["employee_id"],
            "check_out",
            datetime(2026, 11, 18, 10, 1, tzinfo=qatar),
            database_path=self.database_path,
        )
        for day in (1, 2):
            database.record_attendance(
                self.employees["Alice"]["employee_id"],
                "check_in",
                datetime(2026, 10, day, 9, 5, tzinfo=qatar),
                database_path=self.database_path,
            )
            database.record_attendance(
                self.employees["Alice"]["employee_id"],
                "check_out",
                datetime(2026, 10, day, 18, 0, tzinfo=qatar),
                database_path=self.database_path,
            )
        database.record_attendance(
            self.employees["Alice"]["employee_id"],
            "check_in",
            datetime(2026, 11, 5, 9, 5, tzinfo=qatar),
            database_path=self.database_path,
        )
        database.record_attendance(
            self.employees["Alice"]["employee_id"],
            "check_out",
            datetime(2026, 11, 5, 18, 0, tzinfo=qatar),
            database_path=self.database_path,
        )

        answer = self.ask(
            "Is attendance improving compared with last month?"
        )["answer"]
        self.assertIn("2026-11-01–2026-11-17", answer)
        self.assertIn("2026-10-01–2026-10-17", answer)
        self.assertIn("not consistently improving", answer)
        self.assertIn("not a scheduled-workday absence rate", answer)

    def test_chatbot_shows_open_checkouts_and_highlights_daily_issues(self):
        open_checkins = self.ask("Show employees who have not checked out.")["answer"]
        self.assertIn("Alice", open_checkins)
        self.assertNotIn("Bob", open_checkins)

        summary = self.ask(
            "Give me a summary of today's attendance and highlight any issues."
        )["answer"]
        self.assertIn(
            "Issues to review: 1 absent; 1 employee(s) have not checked out",
            summary,
        )
        self.assertNotIn("late arrival(s)", summary)

    def test_chat_history_is_scoped_and_clear_resets_context(self):
        photos = [{
            "employee_name": "Alice",
            "action": "check_in",
            "timestamp": "2026-11-18T09:05:00+03:00",
            "url": "/attendance/record/1/photo/check_in",
        }]
        database.save_chatbot_turn(
            "session-a", "Who is absent?", "Carol", {"intent": "absent"},
            self.database_path,
            photos=photos,
        )
        database.save_chatbot_turn(
            "session-b", "Who is late?", "Alice", {"intent": "late"},
            self.database_path,
        )
        self.assertEqual(len(database.chatbot_history("session-a", self.database_path)), 2)
        self.assertEqual(
            database.chatbot_history("session-a", self.database_path)[1]["photos"],
            photos,
        )
        self.assertEqual(len(database.chatbot_history("session-b", self.database_path)), 2)
        self.assertEqual(
            database.chatbot_context("session-a", self.database_path)["intent"],
            "absent",
        )
        database.clear_chatbot_history("session-a", self.database_path)
        self.assertEqual(database.chatbot_history("session-a", self.database_path), [])
        self.assertEqual(database.chatbot_context("session-a", self.database_path), {})
        self.assertEqual(len(database.chatbot_history("session-b", self.database_path)), 2)


if __name__ == "__main__":
    unittest.main()
