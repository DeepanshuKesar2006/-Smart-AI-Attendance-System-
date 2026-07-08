import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backup import BackupManager


def main():
    manager = BackupManager()
    path = manager.create_backup()
    print(f"Backup created at: {path}")


if __name__ == "__main__":
    main()