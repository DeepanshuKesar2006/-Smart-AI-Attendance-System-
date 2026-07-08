import sqlite3
from config.settings import DATABASE_PATH


class DatabaseManager:
    def __init__(self):
        self.connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL
            )
            """
        )

    def execute(self, query, params=()):
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor

    def fetchone(self, query, params=()):
        cursor = self.execute(query, params)
        return cursor.fetchone()

    def fetchall(self, query, params=()):
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def close(self):
        self.connection.close()