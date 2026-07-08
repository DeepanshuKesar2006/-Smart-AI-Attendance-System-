# Smart AI Attendance System (V3)

A real-time face recognition attendance system built with OpenCV, FaceNet embeddings, and cosine similarity matching, with a Streamlit dashboard for analytics and student management.

## Features

- Real-time face detection (Haar Cascade) and recognition (FaceNet embeddings)
- Cosine similarity matching with configurable confidence threshold
- Unknown face detection
- Attendance marking with cooldown (prevents duplicate marks per session)
- SQLite database for attendance records
- CSV export of attendance history
- Streamlit dashboard: live attendance, historical analytics, student management
- Webcam-based student registration with image quality checks (blur, brightness, face size)
- Duplicate face detection during registration
- Backup and restore (database + embeddings + dataset)
- Centralized logging

## Tech Stack

- Python
- OpenCV (face detection, camera handling)
- keras-facenet (face embeddings)
- scikit-learn / NumPy (cosine similarity)
- SQLite (attendance storage)
- Streamlit (dashboard)
- Pandas (analytics)

## Project Structure

\`\`\`
FaceAttendanceSystem_V3/
├── run.py                  # Main recognition + attendance loop
├── app/                    # Core logic (camera, detection, embeddings, matching, attendance, db)
├── config/settings.py      # Central configuration
├── utils/                  # Logger, image quality checks, helpers, exceptions
├── dashboard/dashboard.py  # Streamlit dashboard
├── scripts/                # CLI utilities (register, delete, backup, export)
├── dataset/                # Student face images (not committed)
├── embeddings/             # Generated face embeddings (not committed)
└── database/               # SQLite attendance database (not committed)
\`\`\`

## Setup

\`\`\`bash
git clone <your-repo-url>
cd FaceAttendanceSystem_V3
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## Usage

### 1. Register students

If you already have a dataset of images organized as \`dataset/<student_name>/*.jpg\`:

\`\`\`bash
python scripts/build_embeddings_from_dataset.py
\`\`\`

To register a new student live via webcam:

\`\`\`bash
python scripts/register_student.py
\`\`\`

### 2. Run attendance recognition

\`\`\`bash
python run.py
\`\`\`

Press \`q\` to quit.

### 3. View the dashboard

\`\`\`bash
streamlit run dashboard/dashboard.py
\`\`\`

### 4. Other utilities

\`\`\`bash
python scripts/export_attendance.py     # Export attendance history to CSV
python scripts/create_backup.py         # Backup database + embeddings + dataset
python scripts/restore_backup.py        # Restore from a backup
python scripts/delete_student.py        # Remove a student completely
\`\`\`

## Configuration

All tunable parameters (similarity threshold, cooldown duration, camera resolution, detection sensitivity) live in \`config/settings.py\`.

## How It Works

1. **Detection** — Haar Cascade locates faces in each webcam frame.
2. **Embedding** — Each detected face is passed through a FaceNet model, producing a 128-dimension vector representation.
3. **Matching** — The new embedding is compared via cosine similarity against stored embeddings for each registered student. The closest match above the similarity threshold is accepted; otherwise the face is labeled "Unknown."
4. **Attendance** — On a confident match, attendance is marked in SQLite, respecting a per-student cooldown so the same person isn't re-marked every frame.

## Limitations

- Recognition accuracy depends on webcam quality, lighting, and the number/variety of training images per student.
- Not designed for large-scale deployment (hundreds of students) without moving from a linear embedding scan to an indexed similarity search (e.g. FAISS).
- No liveness detection — a photo of a registered student could currently be recognized as that student.