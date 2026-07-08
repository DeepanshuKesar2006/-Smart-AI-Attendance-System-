from collections import Counter
from datetime import datetime, timedelta

from app.attendance import AttendanceManager


class ReportGenerator:
    def __init__(self):
        self.attendance = AttendanceManager()

    def weekly_summary(self):
        """Returns {student_name: days_present} for the last 7 days."""
        all_records = self.attendance.get_all_attendance()
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        recent = [r for r in all_records if r[1] >= cutoff]
        counts = Counter(r[0] for r in recent)
        return dict(counts)

    def student_attendance_rate(self, student_name, total_days):
        """Returns attendance percentage for a student over total_days."""
        all_records = self.attendance.get_all_attendance()
        days_present = len({r[1] for r in all_records if r[0] == student_name})

        if total_days == 0:
            return 0.0
        return round((days_present / total_days) * 100, 1)