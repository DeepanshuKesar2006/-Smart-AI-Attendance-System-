import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.registration import RegistrationManager
from app.database import DatabaseManager
from config.settings import DATASET_DIR
from utils.logger import logger


def main():
    manager = RegistrationManager()
    known = list(manager.embeddings_db.keys())

    if not known:
        print("No students registered.")
        return

    print("Registered students:")
    for i, name in enumerate(known, 1):
        print(f"  {i}. {name}")

    student_name = input("\nEnter exact name to delete: ").strip()

    if student_name not in manager.embeddings_db:
        print("Student not found.")
        return

    confirm = input(f"This will delete '{student_name}' embeddings, dataset images, "
                     f"and attendance history. Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    # Remove embeddings
    del manager.embeddings_db[student_name]
    manager._save_embeddings()

    # Remove dataset folder
    folder_path = os.path.join(DATASET_DIR, student_name)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    # Remove attendance history
    db = DatabaseManager()
    db.execute("DELETE FROM attendance WHERE student_name = ?", (student_name,))

    logger.info(f"Deleted student: {student_name}")
    print(f"Deleted '{student_name}' completely.")


if __name__ == "__main__":
    main()