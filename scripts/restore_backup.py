import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backup import BackupManager


def main():
    manager = BackupManager()
    backups = manager.list_backups()

    if not backups:
        print("No backups found.")
        return

    print("Available backups:")
    for i, name in enumerate(backups, 1):
        print(f"  {i}. {name}")

    choice = input("\nEnter number to restore: ").strip()

    try:
        index = int(choice) - 1
        selected = backups[index]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    confirm = input(f"This will overwrite current data with '{selected}'. Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    manager.restore_backup(selected)
    print("Restore complete.")


if __name__ == "__main__":
    main()