import sys
import os
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.attendance import AttendanceManager
from config.settings import BASE_DIR


def main():
    manager = AttendanceManager()
    records = manager.get_all_attendance()

    if not records:
        print("No attendance records found.")
        return

    export_dir = os.path.join(BASE_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)

    filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(export_dir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student Name", "Date", "Time"])
        writer.writerows(records)

    print(f"Exported {len(records)} records to {filepath}")


if __name__ == "__main__":
    main()