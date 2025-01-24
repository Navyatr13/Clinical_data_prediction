import sqlite3

# Path to the database
db_path = r"D:\Clinical_data_prediction\data\sepsis_predictions.db"

# Connect to the database
conn = sqlite3.connect(db_path)

# Create a cursor object
cursor = conn.cursor()

# Query the table
query = "SELECT * FROM predictions"
cursor.execute(query)

# Fetch all rows
rows = cursor.fetchall()

# Print the contents
print("Table Contents:")
for row in rows:
    print(row)

# Close the connection
conn.close()
