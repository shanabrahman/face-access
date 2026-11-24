"""Safe, database-grounded attendance chatbot with local intent handling."""
import calendar
import csv
import io
import re
import textwrap
import zipfile
from datetime import date, datetime, time as datetime_time, timedelta
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import database as attendance_db


SUGGESTIONS = [
    "Who is absent today?",
    "Who all are present today?",
    "Who all checked out today?",
    "Which five employees have the most late arrivals this month?",
    "Which department has the highest absenteeism today?",
    "Is attendance improving compared with last month?",
    "Show employees who have not checked out.",
    "Summarize today's attendance and highlight any issues.",
    "Generate today's attendance report.",
]

MONTHS = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name})


def _period(question, today, previous):
    query = question.casefold()
    if "yesterday" in query:
        selected = today - timedelta(days=1)
        return selected, selected
    if "last week" in query:
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=6)
    if "this week" in query:
        start = today - timedelta(days=today.weekday())
        return start, today
    if "last 7 days" in query or "past 7 days" in query:
        return today - timedelta(days=6), today
    if "last 30 days" in query or "past 30 days" in query:
        return today - timedelta(days=29), today
    if "last month" in query or "previous month" in query:
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    if "this month" in query or "monthly" in query or "this month" in query:
        return today.replace(day=1), today
    month_day = re.search(
        r"\b(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?\b", query
    )
    if month_day and month_day.group(2) in MONTHS:
        day = int(month_day.group(1))
        month = MONTHS[month_day.group(2)]
        year = int(month_day.group(3) or today.year)
        try:
            selected = date(year, month, day)
        except ValueError:
            return None, None
        if not month_day.group(3) and selected > today:
            selected = selected.replace(year=selected.year - 1)
        return selected, selected
    follow_up = bool(re.search(r"\b(what about|and|how about)\b", query))
    if (
        follow_up
        and previous.get("start_date")
        and previous.get("end_date")
    ):
        start = date.fromisoformat(previous["start_date"])
        end = date.fromisoformat(previous["end_date"])
        if start == end and any(word in query for word in ("week", "month", "7 days", "30 days")):
            return _period(query, today, {})
        return start, end
    return today, today


def _classify(question, employees, previous):
    query = re.sub(r"\s+", " ", question.casefold()).strip()
    names = sorted(
        (employee["full_name"] for employee in employees),
        key=len,
        reverse=True,
    )
    employee_name = next(
        (name for name in names if re.search(
            rf"(?<![\w]){re.escape(name.casefold())}(?![\w])", query
        )),
        None,
    )
    departments = sorted(
        {employee["department"] for employee in employees if employee["department"]},
        key=len,
        reverse=True,
    )
    department = next(
        (name for name in departments if re.search(
            rf"(?<![\w]){re.escape(name.casefold())}(?![\w])", query
        )),
        None,
    )
    is_follow_up = bool(re.search(r"\b(what about|and|how about)\b", query))
    if not employee_name and previous.get("employee_name") and is_follow_up:
        employee_name = previous["employee_name"]
    if not department and is_follow_up:
        department = previous.get("department")

    if any(word in query for word in ("report", "download", "export")):
        intent = "report"
    elif employee_name and re.search(
        r"\b(is|was|did|when|what time|attendance status)\b", query
    ):
        intent = "employee"
    elif employee_name and is_follow_up and previous.get("intent"):
        intent = previous["intent"]
    elif any(word in query for word in ("absenteeism", "absence rate", "absent rate")):
        intent = "department_absenteeism"
    elif any(word in query for word in ("absent", "absence")):
        intent = "absent"
    elif any(word in query for word in ("late", "tardy", "after 9", "after 10")):
        intent = "late"
    elif any(word in query for word in ("on time", "punctual", "before 9")):
        intent = "on_time"
    elif (
        re.search(
            r"\b(check[\s-]?out|checkout|checked[\s-]?out|clock[\s-]?out|clocked[\s-]?out|signed[\s-]?out)\b",
            query,
        )
        and not re.search(
            r"\b(not|never|yet|haven't|hasn't|have not|has not)\s+"
            r"(?:been\s+)?(?:checked[\s-]?out|clocked[\s-]?out|signed[\s-]?out)\b",
            query,
        )
    ):
        intent = "checked_out"
    elif any(word in query for word in ("on leave", "approved leave", "leave today")):
        intent = "leave"
    elif any(word in query for word in (
        "currently working", "working now", "not checked out", "forgot to check out",
        "not yet checked out", "not clocked out", "not signed out",
        "haven't checked out", "hasn't checked out", "have not checked out",
        "has not checked out", "open check-in", "open check in",
        "who is working", "working today",
    )):
        intent = "working"
    elif any(word in query for word in ("compare", "improving", "trend")) and any(
        word in query for word in ("week", "month", "last 7 days", "last 30 days")
    ):
        intent = "compare"
    elif (
        not employee_name
        and re.match(r"^(is|was|did)\s+\S+", query)
        and not re.search(
            r"\b(anyone|anybody|everyone|everybody|someone|somebody|"
            r"all staff|any staff|all employees|any employee|team)\b",
            query,
        )
    ):
        intent = "employee"
    elif any(word in query for word in ("present", "attendance status")) and not department:
        intent = "present"
    elif any(word in query for word in ("average working hours", "average hours", "overtime", "worked more", "worked less")):
        intent = "hours"
    elif department or any(word in query for word in ("department", "compare", "highest attendance", "lowest attendance")):
        intent = "department"
    elif any(word in query for word in ("summary", "insight", "analytics", "how many employees")) and not department:
        intent = "summary"
    elif any(word in query for word in (
        "check-in", "check in", "checked in", "check-out", "check out", "records",
    )):
        intent = "employee"
    elif employee_name:
        intent = "employee"
    elif re.search(r"\b(is|was|did|when|attendance for|attendance of)\b", query):
        intent = "employee"
    elif previous.get("intent") and re.search(r"\b(what about|and|how about)\b", query):
        intent = previous["intent"]
    else:
        intent = "unknown"
    return intent, employee_name, department, query


def _employee_status(employee, day, now_local, db, settings):
    shift = employee
    if not shift["shift_id"]:
        return "NOT SCHEDULED", None
    record = db.execute(
        """SELECT a.*, s.name AS record_shift_name, s.start_time, s.end_time,
                  s.grace_period_minutes
           FROM attendance a JOIN shifts s ON s.shift_id = a.shift_id
           WHERE a.employee_id = ? AND a.attendance_date IN (?, ?)
           ORDER BY a.attendance_date DESC, a.check_in DESC""",
        (employee["employee_id"], day.isoformat(), (day - timedelta(days=1)).isoformat()),
    ).fetchall()
    selected = None
    for candidate in record:
        start_time = datetime_time.fromisoformat(candidate["start_time"])
        end_time = datetime_time.fromisoformat(candidate["end_time"])
        overnight = end_time <= start_time
        checkin_date = date.fromisoformat(candidate["attendance_date"])
        checkin = datetime.fromisoformat(candidate["check_in"])
        checkout_date = (
            datetime.fromisoformat(candidate["check_out"]).date()
            if candidate["check_out"] else None
        )
        if checkin_date == day or checkout_date == day:
            selected = candidate
            break
        if (
            overnight and checkin_date == day - timedelta(days=1)
            and candidate["check_out"] is None and day == now_local.date()
        ):
            selected = candidate
            break
    if selected:
        selected = dict(selected)
        selected.update(attendance_db.classify_attendance(selected))
        if selected["check_out"] is None and day == now_local.date():
            return f"{selected['attendance_status']} - CURRENTLY WORKING", selected
        if selected["check_out"] is None:
            return f"{selected['attendance_status']} - NO CHECK-OUT", selected
        return f"{selected['attendance_status']} - CHECKED OUT", selected
    if employee["approved_leave"]:
        return "ON LEAVE", None
    if day in attendance_db._holiday_dates(settings):
        return "HOLIDAY", None
    weekday = attendance_db.WEEKDAYS[day.weekday()]
    if weekday not in attendance_db._working_days(settings):
        return "WEEKLY OFF", None
    if not shift["shift_active"]:
        return "NOT SCHEDULED", None
    if day > now_local.date():
        return "NOT SCHEDULED", None
    if day == now_local.date():
        shift_start = datetime_time.fromisoformat(employee["start_time"])
        deadline = datetime.combine(day, shift_start, now_local.tzinfo) + timedelta(
            minutes=employee["grace_period_minutes"]
        )
        if now_local <= deadline:
            return "NOT CHECKED IN YET", None
    return "ABSENT", None


def _load_employees(db, day):
    rows = db.execute(
        """SELECT e.employee_id, e.full_name, e.department, e.status, e.shift_id,
                  s.name AS shift_name, s.start_time, s.end_time,
                  s.grace_period_minutes, s.active AS shift_active,
                  EXISTS (
                    SELECT 1 FROM leave_requests l
                    WHERE l.employee_id = e.employee_id AND l.status = 'approved'
                      AND l.start_date <= ? AND l.end_date >= ?
                  ) AS approved_leave
           FROM employees e LEFT JOIN shifts s ON s.shift_id = e.shift_id
           WHERE e.status = 'active' ORDER BY e.full_name COLLATE NOCASE""",
        (day.isoformat(), day.isoformat()),
    )
    return [dict(row) for row in rows]


def _attendance_rows(db, start, end, department=None, name=None):
    clauses = ["a.attendance_date BETWEEN ? AND ?"]
    params = [start.isoformat(), end.isoformat()]
    if department:
        clauses.append("e.department = ?")
        params.append(department)
    if name:
        clauses.append("e.full_name = ?")
        params.append(name)
    rows = [
        dict(row) for row in db.execute(
            """SELECT a.attendance_id, a.employee_id, e.full_name, e.department,
                      a.attendance_date, a.check_in, a.check_out, a.working_hours,
                      a.late, a.late_after_grace, a.early_departure, a.status,
                      s.name AS shift_name,
                              s.start_time, s.end_time, s.grace_period_minutes,
                              s.standard_work_hours
               FROM attendance a JOIN employees e ON e.employee_id = a.employee_id
               JOIN shifts s ON s.shift_id = a.shift_id
               WHERE """ + " AND ".join(clauses) +
            " ORDER BY a.attendance_date DESC, a.check_in DESC",
            params,
        )
    ]
    for row in rows:
        row.update(attendance_db.classify_attendance(row))
    return rows


def _time_label(stamp):
    if not stamp:
        return "Not yet"
    return datetime.fromisoformat(stamp).strftime("%I:%M %p")


def _status_text(employee, status, record):
    parts = [
        f"{employee['full_name']} — {status}",
        f"Employee ID: {employee['employee_id']}",
        f"Department: {employee['department'] or 'Unassigned'}",
    ]
    if record:
        if record["record_shift_name"]:
            parts.append(f"Shift: {record['record_shift_name']}")
        parts.append(f"Check-in: {_time_label(record['check_in'])}")
        parts.append(f"Check-out: {_time_label(record['check_out'])}")
        if record["check_out"] and record["working_hours"] is not None:
            parts.append(
                f"Completed hours: {record['working_hours']:.2f} "
                "(first check-in to final checkout, including breaks)"
            )
            if record["early_departure"]:
                parts.append(
                    "EARLY DEPARTURE: "
                    f"{record['early_departure_minutes']} minute(s) before shift end"
                )
        elif record["check_out"] is None:
            parts.append(
                "Working hours are not final until check-out. They use first check-in "
                "through final check-out, including breaks."
            )
    elif employee.get("shift_name"):
        parts.append(f"Shift: {employee['shift_name']}")
    return "\n".join(parts)


def answer_question(question, previous_context=None, database_path=attendance_db.DATABASE_FILE, now=None):
    previous_context = previous_context or {}
    settings = attendance_db.settings(database_path)
    timezone = ZoneInfo(settings["timezone"])
    now_local = now.astimezone(timezone) if now and now.tzinfo else (
        now.replace(tzinfo=timezone) if now else datetime.now(timezone)
    )
    with attendance_db.connection(database_path) as db:
        employees = _load_employees(db, now_local.date())
        intent, employee_name, department, query = _classify(
            question, employees, previous_context
        )
        start, end = _period(question, now_local.date(), previous_context)
        if start is None:
            return {
                "answer": "I couldn't understand that date. Please use a valid date such as 10 November 2026.",
                "context": previous_context,
                "report": None,
                "status": None,
            }
        by_name = {item["full_name"]: item for item in employees}
        selected_employee = by_name.get(employee_name)
        if employee_name and selected_employee is None:
            return {
                "answer": f"I couldn't find an active employee named {employee_name} in the employee database.",
                "context": previous_context,
                "report": None,
                "status": None,
            }
        if not employee_name and previous_context.get("employee_name"):
            employee_name = previous_context["employee_name"]
            selected_employee = by_name.get(employee_name)
        rows = _attendance_rows(db, start, end, department, employee_name)
        active_employees = _load_employees(db, now_local.date())
        day_employees = _load_employees(db, start) if start == end else []
        statuses = []
        if start == end:
            for employee in day_employees:
                status, record = _employee_status(
                    employee, start, now_local, db, settings
                )
                statuses.append((employee, status, record))
        matched = []
        context = {
            "intent": intent,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "employee_name": employee_name,
            "department": department or previous_context.get("department"),
        }
        status_indicator = None

        if intent == "employee":
            if selected_employee:
                if start != end:
                    answer = (
                        f"Attendance for {employee_name} from {start.isoformat()} through "
                        f"{end.isoformat()} ({len(rows)} record(s)):\n"
                        + ("\n".join(
                            f"- {row['attendance_date']}: check-in {_time_label(row['check_in'])}, "
                            f"check-out {_time_label(row['check_out'])}"
                            + (
                                f", {row['working_hours']:.2f} completed hours"
                                if row["working_hours"] is not None
                                else ", hours not final"
                            )
                            + (
                                f", EARLY DEPARTURE ({row['early_departure_minutes']} min)"
                                if row["early_departure"] else ""
                            )
                            for row in rows
                        ) if rows else "No attendance records found.")
                    )
                else:
                    status, record = _employee_status(
                        selected_employee, start, now_local, db, settings
                    )
                    status_indicator = status
                    answer = _status_text(selected_employee, status, record)
            elif employee_name is None and re.search(
                r"\b(is|was|did|when|attendance for|attendance of)\b", query
            ):
                answer = (
                    "I couldn't find that active employee. Please check the spelling "
                    "or use the employee's name as shown in Employee Management."
                )
            else:
                answer = (
                    f"Attendance records from {start.isoformat()} through {end.isoformat()} "
                    f"({len(rows)}):\n"
                    + ("\n".join(
                        f"- {row['full_name']} ({row['department'] or 'Unassigned'}), "
                        f"{row['attendance_date']}: check-in {_time_label(row['check_in'])}, "
                        f"check-out {_time_label(row['check_out'])}"
                        + (
                            f", {row['working_hours']:.2f} completed hours"
                            if row["working_hours"] is not None
                            else ", hours not final"
                        )
                        + (
                            f", EARLY DEPARTURE ({row['early_departure_minutes']} min)"
                            if row["early_departure"] else ""
                        )
                        for row in rows[:50]
                    ) if rows else "None.")
                )
        elif intent in {
            "absent", "late", "on_time", "working", "present", "checked_out",
            "leave", "department_absenteeism",
        }:
            target_statuses = {
                "absent": {"ABSENT"},
                "late": {"LATE", "LATE - CURRENTLY WORKING", "LATE - CHECKED OUT"},
                "on_time": {
                    "ON_TIME", "ON_TIME - CURRENTLY WORKING", "ON_TIME - CHECKED OUT",
                },
                "working": {
                    "CURRENTLY WORKING", "ON_TIME - CURRENTLY WORKING",
                    "GRACE_PERIOD - CURRENTLY WORKING", "LATE - CURRENTLY WORKING",
                    "ON_TIME - NO CHECK-OUT", "GRACE_PERIOD - NO CHECK-OUT",
                    "LATE - NO CHECK-OUT",
                },
                "present": {
                    "ON_TIME", "GRACE_PERIOD", "LATE",
                    "ON_TIME - CURRENTLY WORKING", "GRACE_PERIOD - CURRENTLY WORKING",
                    "LATE - CURRENTLY WORKING", "ON_TIME - CHECKED OUT",
                    "GRACE_PERIOD - CHECKED OUT", "LATE - CHECKED OUT",
                    "ON_TIME - NO CHECK-OUT", "GRACE_PERIOD - NO CHECK-OUT",
                    "LATE - NO CHECK-OUT",
                },
                "checked_out": {
                    "ON_TIME - CHECKED OUT", "GRACE_PERIOD - CHECKED OUT",
                    "LATE - CHECKED OUT",
                },
                "leave": {"ON LEAVE"},
                "department_absenteeism": set(),
            }[intent]
            matched = [
                (employee, status, record) for employee, status, record in statuses
                if (
                    (intent == "late" and record and record["attendance_status"] == "LATE")
                    or (intent == "on_time" and record and record["attendance_status"] == "ON_TIME")
                    or (intent == "working" and record and not record["check_out"])
                    or (intent == "present" and record)
                    or (intent == "checked_out" and record and record["check_out"])
                    or (intent == "leave" and status == "ON LEAVE")
                    or (intent == "absent" and status in target_statuses)
                )
            ]
            if employee_name:
                matched = [
                    item for item in matched
                    if item[0]["full_name"] == employee_name
                ]
            labels = {
                "absent": "Absent",
                "late": "Late arrivals (after the configured grace period)",
                "on_time": "Present on time",
                "working": "Currently working / not checked out",
                "present": "Present",
                "checked_out": "Checked out",
                "leave": "On leave",
                "department_absenteeism": "Department absenteeism",
            }
            if intent == "department_absenteeism" and start != end:
                answer = (
                    "I can't reliably compare historical absenteeism because weekly-off "
                    "and public-holiday schedules are not configured. Ask for today's "
                    "department absenteeism or configure the work calendar first."
                )
            elif intent == "department_absenteeism":
                department_totals = {}
                for employee, status, record in statuses:
                    if status != "ABSENT" and record is None:
                        continue
                    name = employee["department"] or "Unassigned"
                    totals = department_totals.setdefault(name, {"absent": 0, "eligible": 0})
                    totals["eligible"] += 1
                    totals["absent"] += int(status == "ABSENT")
                if department:
                    department_totals = {
                        department: department_totals[department]
                    } if department in department_totals else {}
                ranked = sorted(
                    department_totals.items(),
                    key=lambda item: (
                        -(item[1]["absent"] / item[1]["eligible"]),
                        -item[1]["absent"],
                        item[0].casefold(),
                    ),
                )
                if not ranked:
                    answer = (
                        "No departments have employees whose shifts are due for attendance "
                        f"today ({start.isoformat()})."
                    )
                else:
                    ranking = "\n".join(
                        f"- {name}: {totals['absent']} absent of {totals['eligible']} "
                        f"scheduled ({100 * totals['absent'] / totals['eligible']:.1f}%)"
                        for name, totals in ranked
                    )
                    top_name, top_totals = ranked[0]
                    answer = (
                        f"Highest absenteeism today: {top_name} "
                        f"({top_totals['absent']} of {top_totals['eligible']} scheduled, "
                        f"{100 * top_totals['absent'] / top_totals['eligible']:.1f}%).\n"
                        f"Department rates for {start.isoformat()}:\n{ranking}\n"
                        "Only employees whose shift grace deadline has passed are counted; "
                        "approved leave and shifts not yet due are excluded. Workday and "
                        "holiday calendars are not configured."
                    )
            elif start != end and intent == "leave":
                leave_rows = db.execute(
                    """SELECT e.employee_id, e.full_name, e.department,
                              l.start_date, l.end_date
                       FROM leave_requests l JOIN employees e
                         ON e.employee_id = l.employee_id
                       WHERE l.status = 'approved' AND l.start_date <= ?
                         AND l.end_date >= ? AND e.status = 'active'
                         ORDER BY e.full_name COLLATE NOCASE""",
                    (end.isoformat(), start.isoformat()),
                ).fetchall()
                if department:
                    leave_rows = [
                        row for row in leave_rows if row["department"] == department
                    ]
                answer = (
                    f"Approved leave overlapping {start.isoformat()} through "
                    f"{end.isoformat()} ({len(leave_rows)}):\n"
                    + ("\n".join(
                        f"- {row['full_name']} ({row['department'] or 'Unassigned'}), "
                        f"{row['start_date']} to {row['end_date']}"
                        for row in leave_rows
                    ) if leave_rows else "None.")
                )
            elif start != end and intent == "absent":
                answer = (
                    "I can't reliably list historical absences for a date range because "
                    "weekly-off and public-holiday schedules are not configured. "
                    "Configure the work calendar before using multi-day absence totals."
                )
            elif start != end:
                matched_names = {row["full_name"] for row in rows}
                if intent == "late":
                    rows = [row for row in rows if row["attendance_status"] == "LATE"]
                    matched_names = {row["full_name"] for row in rows}
                elif intent == "on_time":
                    total_attendance = len(rows)
                    rows = [row for row in rows if row["attendance_status"] == "ON_TIME"]
                    matched_names = {row["full_name"] for row in rows}
                elif intent == "checked_out":
                    rows = [row for row in rows if row["check_out"]]
                    matched_names = {row["full_name"] for row in rows}
                if intent == "late" and employee_name and "days" in query:
                    answer = (
                        f"{employee_name} had {len({row['attendance_date'] for row in rows})} "
                        f"recorded late day(s) from {start.isoformat()} through {end.isoformat()}."
                    )
                elif intent == "on_time" and any(
                    word in query for word in ("percentage", "percent")
                ):
                    percentage = (
                        100 * len(rows) / total_attendance
                        if total_attendance else None
                    )
                    answer = (
                        f"{percentage:.1f}% of recorded attendance events were on time "
                        f"({len(rows)} of {total_attendance}) from {start.isoformat()} "
                        f"through {end.isoformat()}."
                        if percentage is not None else
                        f"No attendance records were found from {start.isoformat()} through {end.isoformat()}."
                    )
                elif intent == "late" and any(
                    word in query for word in ("who", "show", "list", "top", "which", "most")
                ):
                    if re.search(r"\b(top|most|highest|rank|five|5)\b", query):
                        totals = {}
                        for row in rows:
                            employee_totals = totals.setdefault(
                                row["employee_id"],
                                {
                                    "name": row["full_name"],
                                    "department": row["department"] or "Unassigned",
                                    "late": 0,
                                    "after_grace": 0,
                                },
                            )
                            employee_totals["late"] += 1
                            employee_totals["after_grace"] += int(
                                row["late_after_grace"]
                            )
                        leaders = sorted(
                            totals.values(),
                            key=lambda item: (-item["late"], item["name"].casefold()),
                        )[:5]
                        answer = (
                            f"Employees with the most recorded late arrivals from "
                            f"{start.isoformat()} through "
                            f"{end.isoformat()}:\n"
                            + (
                                "\n".join(
                                    f"- {item['name']} ({item['department']}): "
                                    f"{item['late']} late arrival(s), "
                                    f"{item['after_grace']} after grace"
                                    for item in leaders
                                )
                                if leaders else "None."
                            )
                            + "\nLate arrivals are check-ins beyond the assigned shift's "
                            "grace deadline; the after-grace count is shown separately."
                        )
                    else:
                        answer = (
                            f"{labels[intent]} from {start.isoformat()} through {end.isoformat()} "
                            f"({len(rows)} record(s)):\n"
                            + ("\n".join(
                                f"- {row['full_name']} ({row['department'] or 'Unassigned'}), "
                                f"{row['attendance_date']} at {_time_label(row['check_in'])}, "
                                f"{_late_description(row)}"
                                for row in rows[:20]
                            ) if rows else "None.")
                        )
                else:
                    answer = (
                        f"{labels[intent]} from {start.isoformat()} through {end.isoformat()}: "
                        f"{len(rows)} attendance record(s), {len(matched_names)} employee(s)."
                    )
            else:
                if intent == "on_time" and any(
                    word in query for word in ("percentage", "percent")
                ):
                    total_attendance = sum(1 for _, _, record in statuses if record)
                    percentage = (
                        100 * len(matched) / total_attendance
                        if total_attendance else None
                    )
                    answer = (
                        f"{percentage:.1f}% of recorded attendance events were on time "
                        f"({len(matched)} of {total_attendance}) on {start.isoformat()}."
                        if percentage is not None else
                        f"No attendance records were found on {start.isoformat()}."
                    )
                else:
                    answer = f"{labels[intent]} on {start.isoformat()} ({len(matched)}):"
                    answer += "\n" + ("\n".join(
                    f"- {employee['employee_id']} — {employee['full_name']} "
                    f"({employee['department'] or 'Unassigned'})"
                    + (f", check-in {_time_label(record['check_in'])}" if record else "")
                    + (
                        f", check-out {_time_label(record['check_out'])}"
                        if intent == "checked_out" and record else ""
                    )
                    + (
                        f", {_late_description(record)}"
                        if intent == "late" and record else ""
                    )
                    for employee, _, record in matched
                        ) if matched else "None.")
                if intent == "absent":
                    answer += (
                        "\nAbsence is assessed after each assigned shift's start plus grace period. "
                        "Weekly-off and public-holiday calendars are not configured in this app."
                    )
                if intent == "late":
                    answer += (
                        "\nA late arrival is a check-in after the assigned shift's grace "
                        "deadline. Arrivals within grace are classified separately."
                    )
            if department and intent != "department_absenteeism":
                matched = [
                    item for item in matched
                    if item[0]["department"] == department
                ]
                if start == end:
                    answer = f"{labels[intent]} in {department} on {start.isoformat()} ({len(matched)}):"
                    answer += "\n" + ("\n".join(
                        f"- {employee['employee_id']} — {employee['full_name']}"
                        + (f", check-in {_time_label(record['check_in'])}" if record else "")
                        for employee, _, record in matched
                    ) if matched else "None.")
        elif intent == "summary":
            counts = {}
            for _, status, _ in statuses:
                counts[status] = counts.get(status, 0) + 1
            completed = [row["working_hours"] for row in rows if row["working_hours"] is not None]
            late_count = sum(1 for row in rows if row["attendance_status"] == "LATE")
            after_grace_count = sum(1 for row in rows if row["late_after_grace"])
            early_departure_count = sum(1 for row in rows if row["early_departure"])
            open_count = sum(1 for row in rows if row["check_out"] is None)
            answer = (
                f"Attendance summary for {start.isoformat()} through {end.isoformat()}:\n"
                f"Current active employees: {len(active_employees)}\n"
                f"Attendance records: {len(rows)}\n"
                f"Recorded late flags: {late_count}\n"
                f"Completed records with hours: {len(completed)}\n"
                f"Average completed hours: {sum(completed) / len(completed):.2f}"
                if completed else
                f"Attendance summary for {start.isoformat()} through {end.isoformat()}:\n"
                f"Current active employees: {len(active_employees)}\n"
                f"Attendance records: {len(rows)}\n"
                f"Recorded late flags: {late_count}\n"
                "Average completed hours: no completed records."
            )
            if start == end:
                answer += "\n" + "\n".join(
                    f"{status}: {count}" for status, count in sorted(counts.items())
                )
                issues = []
                absent_count = counts.get("ABSENT", 0)
                if absent_count:
                    issues.append(f"{absent_count} absent")
                if late_count:
                    issues.append(f"{late_count} late arrival(s)")
                if after_grace_count:
                    issues.append(f"{after_grace_count} check-in(s) after grace")
                if open_count:
                    issues.append(f"{open_count} employee(s) have not checked out")
                if early_departure_count:
                    issues.append(
                        f"{early_departure_count} early departure(s)"
                    )
                answer += "\nIssues to review: " + (
                    "; ".join(issues) if issues else "No attendance issues found."
                )
            else:
                answer += "\nAbsence totals over a range require a configured work calendar."
        elif intent == "hours":
            if selected_employee and start == end:
                status, record = _employee_status(
                    selected_employee, start, now_local, db, settings
                )
                if not record:
                    answer = f"No attendance record was found for {employee_name} on {start.isoformat()}."
                elif not record["check_out"]:
                    answer = f"{employee_name} is still checked in. Working hours are not final until checkout."
                else:
                    answer = (
                        f"{employee_name} completed {record['working_hours']:.2f} hours "
                        f"from first check-in through final check-out on {start.isoformat()}, "
                        "including breaks."
                    )
                    if record["early_departure"]:
                        answer += (
                            f" EARLY DEPARTURE: {record['early_departure_minutes']} "
                            "minute(s) before shift end."
                        )
            else:
                completed = [row for row in rows if row["working_hours"] is not None]
                if "more than" in query or "overtime" in query:
                    overtime = [
                        row for row in completed
                        if row["working_hours"] > row["standard_work_hours"]
                    ]
                    answer = (
                        f"Employees with completed hours above their shift standard "
                        f"from {start.isoformat()} through {end.isoformat()} "
                        f"({len(overtime)} record(s)):\n"
                        + ("\n".join(
                            f"- {row['full_name']}: {row['working_hours']:.2f} hours "
                            f"(standard {row['standard_work_hours']:.2f})"
                            for row in overtime
                        ) if overtime else "None.")
                    )
                    completed = []
                elif "less than" in query:
                    under = [
                        row for row in completed
                        if row["working_hours"] < row["standard_work_hours"]
                    ]
                    answer = (
                        f"Employees with completed hours below their shift standard "
                        f"from {start.isoformat()} through {end.isoformat()} "
                        f"({len(under)} record(s)):\n"
                        + ("\n".join(
                            f"- {row['full_name']}: {row['working_hours']:.2f} hours "
                            f"(standard {row['standard_work_hours']:.2f})"
                            for row in under
                        ) if under else "None.")
                    )
                    completed = []
                average = (
                    sum(row["working_hours"] for row in completed) / len(completed)
                    if completed else None
                )
                if "more than" not in query and "overtime" not in query and "less than" not in query:
                    answer = (
                        f"Completed work records: {len(completed)} from {start.isoformat()} "
                        f"through {end.isoformat()}.\n"
                        f"Average completed hours: {average:.2f}"
                        if average is not None else
                        f"No completed-hours records were found from {start.isoformat()} through {end.isoformat()}."
                    )
                if any(row["check_out"] is None for row in rows):
                    answer += "\nOpen check-ins are excluded; their working hours are not final."
        elif intent == "compare":
            if "month" in query:
                today = now_local.date()
                current_start = today.replace(day=1)
                current_end = today - timedelta(days=1)
                previous_month_end = current_start - timedelta(days=1)
                previous_start = previous_month_end.replace(day=1)
                elapsed_days = max(0, (current_end - current_start).days + 1)
                previous_end = min(
                    previous_start + timedelta(days=max(elapsed_days, 1) - 1),
                    previous_month_end,
                )
            else:
                current_start = now_local.date() - timedelta(days=now_local.weekday())
                current_end = now_local.date()
                previous_end = current_start - timedelta(days=1)
                previous_start = previous_end - timedelta(days=6)
            current_rows = _attendance_rows(
                db, current_start, current_end, department, employee_name
            )
            previous_rows = _attendance_rows(
                db, previous_start, previous_end, department, employee_name
            )
            def period_summary(period_rows):
                completed = [
                    row["working_hours"] for row in period_rows
                    if row["working_hours"] is not None
                ]
                return (
                    len({row["employee_id"] for row in period_rows}),
                    sum(1 for row in period_rows if row["attendance_status"] == "LATE"),
                    sum(completed) / len(completed) if completed else None,
                )
            current_summary = period_summary(current_rows)
            previous_summary = period_summary(previous_rows)
            if "month" in query:
                current_days = max(0, (current_end - current_start).days + 1)
                previous_days = max(0, (previous_end - previous_start).days + 1)
                current_per_day = len(current_rows) / current_days if current_days else 0
                previous_per_day = len(previous_rows) / previous_days if previous_days else 0
                current_late_rate = (
                    current_summary[1] / len(current_rows) if current_rows else 0
                )
                previous_late_rate = (
                    previous_summary[1] / len(previous_rows) if previous_rows else 0
                )
                if not current_days or not previous_days:
                    assessment = "There are not enough completed days to compare yet."
                elif not current_rows and not previous_rows:
                    assessment = "There are no recorded check-ins in either period."
                elif (
                    current_per_day > previous_per_day
                    and current_late_rate <= previous_late_rate
                ):
                    assessment = "Recorded attendance is improving: check-ins per calendar day increased and the late-arrival share did not increase."
                elif (
                    current_per_day < previous_per_day
                    or current_late_rate > previous_late_rate
                ):
                    assessment = "Recorded attendance is not consistently improving: check-in volume or the late-arrival share worsened."
                else:
                    assessment = "Recorded attendance is broadly unchanged across these periods."
                answer = (
                    f"Month-to-date comparison (completed days only):\n"
                    f"This month ({current_start.isoformat()}–{current_end.isoformat()}): "
                    f"{len(current_rows)} check-in(s), {current_per_day:.2f} per calendar day, "
                    f"{current_summary[1]} late flag(s) "
                    f"({100 * current_late_rate:.1f}%).\n"
                    f"Previous month, same number of days "
                    f"({previous_start.isoformat()}–{previous_end.isoformat()}): "
                    f"{len(previous_rows)} check-in(s), {previous_per_day:.2f} per calendar day, "
                    f"{previous_summary[1]} late flag(s) "
                    f"({100 * previous_late_rate:.1f}%).\n"
                    f"Assessment: {assessment}\n"
                    "This uses recorded check-ins per calendar day, not a scheduled-workday "
                    "absence rate; a workday calendar is not configured."
                )
            else:
                current_hours = (
                    f"{current_summary[2]:.2f}" if current_summary[2] is not None else "no completed records"
                )
                previous_hours = (
                    f"{previous_summary[2]:.2f}" if previous_summary[2] is not None else "no completed records"
                )
                answer = (
                    f"Attendance comparison:\n"
                    f"This week ({current_start.isoformat()}–{current_end.isoformat()}): "
                    f"{len(current_rows)} record(s), {current_summary[0]} employee(s), "
                    f"{current_summary[1]} recorded late flags, average completed hours {current_hours}.\n"
                    f"Previous week ({previous_start.isoformat()}–{previous_end.isoformat()}): "
                    f"{len(previous_rows)} record(s), {previous_summary[0]} employee(s), "
                    f"{previous_summary[1]} recorded late flags, average completed hours {previous_hours}."
                )
        elif intent == "department":
            departments = {}
            for row in rows:
                item = departments.setdefault(
                    row["department"] or "",
                    {"department": row["department"] or "", "present": 0,
                     "late_count": 0, "record_count": 0, "on_time_count": 0},
                )
                item["present"] += 1
                item["late_count"] += int(row["attendance_status"] == "LATE")
                item["on_time_count"] += int(row["attendance_status"] == "ON_TIME")
                item["record_count"] += 1
            if department:
                dept_rows = [row for row in departments.values() if row["department"] == department]
                answer = (
                    f"{department}: {dept_rows[0]['present']} employee(s) recorded attendance "
                    f"and {dept_rows[0]['late_count'] or 0} late flag(s) from "
                    f"{start.isoformat()} through {end.isoformat()}."
                    if dept_rows else
                    f"No attendance records were found for {department} from {start.isoformat()} through {end.isoformat()}."
                )
            elif departments:
                department_rows = list(departments.values())
                if "punctual" in query or "on time" in query:
                    department_rows.sort(
                        key=lambda item: (
                            -(item["on_time_count"] / item["record_count"])
                            if item["record_count"] else 0,
                            item["department"] or "",
                        )
                    )
                heading = (
                    "Department punctuality (based on recorded flags)"
                    if "punctual" in query or "on time" in query
                    else "Department attendance"
                )
                answer = f"{heading} from {start.isoformat()} through {end.isoformat()}:\n" + "\n".join(
                    f"- {row['department'] or 'Unassigned'}: {row['present']} present, "
                    f"{row['late_count'] or 0} recorded late flags"
                    + (
                        f", {100 * row['on_time_count'] / row['record_count']:.1f}% on time"
                        if "punctual" in query or "on time" in query else ""
                    )
                    for row in department_rows
                )
                if "highest" in query and department_rows:
                    answer = (
                        f"Highest attendance: {department_rows[0]['department'] or 'Unassigned'} "
                        f"({department_rows[0]['present']} employee(s) recorded attendance).\n"
                        + answer
                    )
                elif ("punctual" in query or "on time" in query) and department_rows:
                    top = department_rows[0]
                    percentage = 100 * top["on_time_count"] / top["record_count"]
                    answer = (
                        f"Best recorded punctuality: {top['department'] or 'Unassigned'} "
                        f"({percentage:.1f}% on time).\n" + answer
                    )
            else:
                answer = f"No department attendance records were found from {start.isoformat()} through {end.isoformat()}."
        elif intent == "report":
            if "absent" in query:
                report_kind = "absent"
            elif "late" in query:
                report_kind = "late"
            else:
                report_kind = "attendance"
            if start != end and report_kind == "absent":
                answer = "Absent-employee reports are available for one date at a time; historical work calendars are not configured."
                report = None
            else:
                report = {
                    "kind": report_kind,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "department": department or "",
                }
                answer = (
                    f"Your {report_kind} report for {start.isoformat()} through {end.isoformat()} "
                    "is ready. Use the download buttons below."
                )
        else:
            answer = (
                "I can answer questions about attendance status, absences, late/on-time records, "
                "check-in/out times, completed hours, departments, summaries, and CSV/Excel/PDF reports. "
                "Please include a date or employee name where needed."
            )
            report = None
        if intent != "report":
            report = None
        photos = []
        photo_action = {
            "present": "check_in",
            "working": "check_in",
            "checked_out": "check_out",
        }.get(intent)
        if photo_action and start == end:
            path_key = (
                "checkin_photo_path" if photo_action == "check_in"
                else "checkout_photo_path"
            )
            timestamp_key = (
                "checkin_photo_timestamp" if photo_action == "check_in"
                else "checkout_photo_timestamp"
            )
            for employee, _, record in matched:
                if not record or (photo_action == "check_out" and not record["check_out"]):
                    continue
                photos.append({
                    "employee_name": employee["full_name"],
                    "attendance_date": start.isoformat(),
                    "action": photo_action,
                    "timestamp": record[timestamp_key],
                    "url": (
                        f"/attendance/record/{record['attendance_id']}/photo/"
                        f"{photo_action}"
                    ) if record[path_key] else None,
                })
        return {
            "answer": answer,
            "context": context,
            "report": report,
            "status": status_indicator,
            "photos": photos,
        }


def report_rows(report, database_path=attendance_db.DATABASE_FILE, now=None):
    start = date.fromisoformat(report["start_date"])
    end = date.fromisoformat(report["end_date"])
    kind = report["kind"]
    department = report.get("department") or None
    with attendance_db.connection(database_path) as db:
        rows = _attendance_rows(db, start, end, department)
        if kind == "late":
            rows = [row for row in rows if row["attendance_status"] == "LATE"]
            return [
                [
                    row["employee_id"], row["full_name"], row["department"],
                    row["attendance_date"], row["check_in"],
                    row["minutes_after_start"], row["grace_minutes_used"],
                    row["late_minutes"], _late_description(row),
                ]
                for row in rows
            ]
        if kind == "absent":
            config = attendance_db.settings(database_path)
            tz = ZoneInfo(config["timezone"])
            local_now = now.astimezone(tz) if now and now.tzinfo else (
                now.replace(tzinfo=tz) if now else datetime.now(tz)
            )
            employees = _load_employees(db, start)
            result = []
            for employee in employees:
                if department and employee["department"] != department:
                    continue
                status, _ = _employee_status(
                    employee, start, local_now, db, config
                )
                if status == "ABSENT":
                    result.append([
                        employee["employee_id"], employee["full_name"],
                        employee["department"], start.isoformat(), status,
                    ])
            return result
        return [
            [
                row["attendance_id"], row["employee_id"], row["full_name"],
                row["department"], row["shift_name"], row["attendance_date"],
                row["check_in"], row["check_out"] or "Not checked out",
                "" if row["working_hours"] is None else row["working_hours"],
                row["attendance_status"],
                "Yes" if row["early_departure"] else "No", row["status"],
            ]
            for row in rows
        ]


def _late_description(row):
    return (
        f"{row['minutes_after_start']} minutes after shift start; "
        f"{row['grace_minutes_used']} grace minutes used; "
        f"{row['late_minutes']} minutes beyond grace"
    )


def build_report(report, file_format="csv", database_path=attendance_db.DATABASE_FILE):
    rows = report_rows(report, database_path)
    headers = {
        "late": [
            "Employee ID", "Employee", "Department", "Date", "Check-in",
            "Minutes after shift start", "Grace minutes used",
            "Minutes beyond grace", "Late details",
        ],
        "absent": ["Employee ID", "Employee", "Department", "Date", "Status"],
        "attendance": [
            "Attendance ID", "Employee ID", "Employee", "Department", "Shift",
            "Date", "Check-in", "Check-out", "Completed hours", "Late", "Early departure", "Status",
        ],
    }[report["kind"]]
    safe_rows = [
        [_safe_cell(value) for value in row]
        for row in rows
    ]
    filename = f"{report['kind']}-attendance-{report['start_date']}-to-{report['end_date']}"
    if file_format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(safe_rows)
        return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", filename + ".csv"
    if file_format == "xlsx":
        return (
            _build_xlsx(headers, safe_rows),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename + ".xlsx",
        )
    if file_format == "pdf":
        pdf, _, _ = _build_pdf(headers, safe_rows)
        return pdf, "application/pdf", filename + ".pdf"
    raise ValueError("Format must be csv, xlsx, or pdf.")


def _safe_cell(value):
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _build_xlsx(headers, rows):
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ElementTree.Element(f"{{{namespace}}}worksheet")
    sheet_data = ElementTree.SubElement(root, f"{{{namespace}}}sheetData")
    for row_index, values in enumerate([headers, *rows], 1):
        row_node = ElementTree.SubElement(
            sheet_data, f"{{{namespace}}}row", {"r": str(row_index)}
        )
        for col_index, value in enumerate(values, 1):
            column = ""
            number = col_index
            while number:
                number, remainder = divmod(number - 1, 26)
                column = chr(65 + remainder) + column
            cell = ElementTree.SubElement(
                row_node, f"{{{namespace}}}c",
                {"r": f"{column}{row_index}", "t": "inlineStr"},
            )
            inline = ElementTree.SubElement(cell, f"{{{namespace}}}is")
            text = ElementTree.SubElement(inline, f"{{{namespace}}}t")
            text.text = str(value)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Attendance" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    package_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", ElementTree.tostring(root, encoding="utf-8"))
    return output.getvalue()


def _build_pdf(headers, rows):
    lines = [" | ".join(headers)]
    lines.extend(" | ".join(str(value) for value in row) for row in rows)
    pages = [lines[index:index + 44] for index in range(0, len(lines), 44)] or [[]]
    objects = []
    page_ids = []
    font_id = 3 + len(pages) * 2
    for index, page_lines in enumerate(pages):
        page_id = 3 + index * 2
        content_id = page_id + 1
        page_ids.append(page_id)
        text_lines = ["BT", "/F1 8 Tf", "30 800 Td", "10 TL"]
        for line in page_lines:
            safe = line.encode("latin-1", "replace").decode("latin-1")
            wrapped = textwrap.wrap(safe, width=110) or [""]
            for segment in wrapped:
                escaped = segment.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                text_lines.append(f"({escaped}) Tj T*")
        text_lines.append("ET")
        stream = "\n".join(text_lines).encode("latin-1")
        objects.extend([
            (page_id, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()),
            (content_id, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"),
        ])
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages_dict = f"<< /Type /Pages /Kids [{' '.join(f'{item} 0 R' for item in page_ids)}] /Count {len(page_ids)} >>".encode()
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects.extend([(1, catalog), (2, pages_dict), (font_id, font)])
    objects.sort()
    output = bytearray(b"%PDF-1.4\n")
    offsets = {0: 0}
    for object_id, body in objects:
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_position = len(output)
    size = font_id + 1
    output.extend(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
    for object_id in range(1, size):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF".encode()
    )
    return bytes(output), "application/pdf", "attendance-report.pdf"
