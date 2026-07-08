from datetime import datetime

from app.database import DatabaseManager
from utils.logger import logger
from config.settings import ATTENDANCE_COOLDOWN_SECONDS


class AttendanceManager:
    def __init__(self):
        self.database = DatabaseManager()
        self.last_marked_at = {}  # student_name -> datetime of last mark, in-memory cache

    def mark_attendance(self, student_name):
        """
        Marks attendance for student_name if not already marked within
        the cooldown window. Returns (success, message).
        """
        if student_name == "Unknown":
            return False, "Unknown face"

        now = datetime.now()

        last_time = self.last_marked_at.get(student_name)
        if last_time is not None:
            elapsed = (now - last_time).total_seconds()
            if elapsed < ATTENDANCE_COOLDOWN_SECONDS:
                return False, "Already marked (cooldown)"

        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Double-check against DB in case of app restart (in-memory cache is empty then)
        existing = self.database.fetchone(
            """
            SELECT * FROM attendance
            WHERE student_name = ? AND attendance_date = ?
            """,
            (student_name, today),
        )

        if existing:
            self.last_marked_at[student_name] = now
            return False, "Already marked today"

        self.database.execute(
            """
            INSERT INTO attendance (student_name, attendance_date, attendance_time)
            VALUES (?, ?, ?)
            """,
            (student_name, today, current_time),
        )

        self.last_marked_at[student_name] = now
        logger.info(f"{student_name} marked present at {current_time}")
        return True, "Attendance marked"

    def get_today_attendance(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return self.database.fetchall(
            "SELECT student_name, attendance_time FROM attendance WHERE attendance_date = ?",
            (today,),
        )

    def get_all_attendance(self):
        return self.database.fetchall(
            "SELECT student_name, attendance_date, attendance_time FROM attendance "
            "ORDER BY attendance_date DESC, attendance_time DESC"
        )