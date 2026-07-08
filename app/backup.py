import os
import shutil
import zipfile
from datetime import datetime

from config.settings import BASE_DIR, DATABASE_PATH, EMBEDDINGS_FILE, DATASET_DIR
from utils.logger import logger

BACKUP_DIR = os.path.join(BASE_DIR, "backup")


class BackupManager:
    def __init__(self):
        os.makedirs(BACKUP_DIR, exist_ok=True)

    def create_backup(self):
        """Zips database, embeddings, and dataset into a timestamped backup file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.zip"
        backup_path = os.path.join(BACKUP_DIR, backup_name)

        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Database
            if os.path.exists(DATABASE_PATH):
                zipf.write(DATABASE_PATH, arcname=os.path.join("database", os.path.basename(DATABASE_PATH)))

            # Embeddings
            if os.path.exists(EMBEDDINGS_FILE):
                zipf.write(EMBEDDINGS_FILE, arcname=os.path.join("embeddings", os.path.basename(EMBEDDINGS_FILE)))

            # Dataset (all student folders)
            if os.path.isdir(DATASET_DIR):
                for root, _, files in os.walk(DATASET_DIR):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.join("dataset", os.path.relpath(full_path, DATASET_DIR))
                        zipf.write(full_path, arcname=arcname)

        logger.info(f"Backup created: {backup_path}")
        return backup_path

    def list_backups(self):
        if not os.path.isdir(BACKUP_DIR):
            return []
        return sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.endswith(".zip")],
            reverse=True,
        )

    def restore_backup(self, backup_filename):
        """Restores database, embeddings, and dataset from a backup zip."""
        backup_path = os.path.join(BACKUP_DIR, backup_filename)

        if not os.path.exists(backup_path):
            logger.warning(f"Backup file not found: {backup_path}")
            return False

        with zipfile.ZipFile(backup_path, "r") as zipf:
            zipf.extractall(BASE_DIR)

        logger.info(f"Restored from backup: {backup_path}")
        return True