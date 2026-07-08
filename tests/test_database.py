from app.database_manager import DatabaseManager

database = DatabaseManager()

rows = database.fetchall(

    "SELECT * FROM attendance"

)

for row in rows:

    print(row)