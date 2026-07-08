import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.attendance import AttendanceManager
from app.registration import RegistrationManager
from config.settings import DATASET_DIR

st.set_page_config(page_title="Smart AI Attendance System", layout="wide")


@st.cache_resource
def get_attendance_manager():
    return AttendanceManager()


@st.cache_resource
def get_registration_manager():
    return RegistrationManager()


attendance_manager = get_attendance_manager()
registration_manager = get_registration_manager()

st.title("📋 Smart AI Attendance System — Dashboard")

tab_today, tab_history, tab_students = st.tabs(
    ["Today's Attendance", "History & Analytics", "Student Management"]
)

# -----------------------------
# Tab 1: Today's Attendance
# -----------------------------
with tab_today:
    st.subheader("Today's Attendance")

    records = attendance_manager.get_today_attendance()

    if records:
        df_today = pd.DataFrame(records, columns=["Student Name", "Time"])
        st.dataframe(df_today, use_container_width=True)
        st.metric("Total marked today", len(df_today))
    else:
        st.info("No attendance marked yet today.")

    if st.button("🔄 Refresh"):
        st.rerun()

# -----------------------------
# Tab 2: History & Analytics
# -----------------------------
with tab_history:
    st.subheader("Full Attendance History")

    all_records = attendance_manager.get_all_attendance()

    if all_records:
        df_all = pd.DataFrame(all_records, columns=["Student Name", "Date", "Time"])
        st.dataframe(df_all, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Attendance count per student**")
            counts = df_all["Student Name"].value_counts()
            st.bar_chart(counts)

        with col2:
            st.markdown("**Attendance per day**")
            daily_counts = df_all.groupby("Date").size()
            st.line_chart(daily_counts)

        csv_data = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full history as CSV",
            data=csv_data,
            file_name="attendance_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No attendance records yet.")

# -----------------------------
# Tab 3: Student Management
# -----------------------------
with tab_students:
    st.subheader("Registered Students")

    known_students = list(registration_manager.embeddings_db.keys())

    if known_students:
        for name in known_students:
            col1, col2 = st.columns([4, 1])
            with col1:
                embedding_count = len(registration_manager.embeddings_db[name])
                st.write(f"**{name}** — {embedding_count} embeddings")
            with col2:
                if st.button("Delete", key=f"delete_{name}"):
                    del registration_manager.embeddings_db[name]
                    registration_manager._save_embeddings()
                    st.success(f"Deleted {name}. Refresh to see changes.")
                    st.rerun()
    else:
        st.info("No students registered yet.")

    st.divider()
    st.markdown(
        "To register a **new** student, run this in your terminal:\n\n"
        "```\npython scripts/register_student.py\n```"
    )