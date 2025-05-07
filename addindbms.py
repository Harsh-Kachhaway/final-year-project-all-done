import sqlite3
from datetime import datetime

conn = sqlite3.connect('data/detection_data.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        plate_text TEXT PRIMARY KEY,
        owner_name TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        message_sent INTEGER DEFAULT 0
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        plate_text TEXT NOT NULL,
        confidence REAL NOT NULL,
        FOREIGN KEY (plate_text) REFERENCES plates(plate_text)
    )
''')

def add_plate(plate_text, owner_name, phone_number):
    cursor.execute('SELECT plate_text FROM plates WHERE plate_text = ?', (plate_text,))
    if cursor.fetchone() is None:
        cursor.execute('''
            INSERT INTO plates (plate_text, owner_name, phone_number, message_sent)
            VALUES (?, ?, ?, 0)
        ''', (plate_text, owner_name, phone_number))
        print(f"[INFO] New plate added: {plate_text}")
    else:
        print(f"[INFO] Plate {plate_text} already exists.")

def add_detection(plate_text, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO detections (timestamp, plate_text, confidence)
        VALUES (?, ?, ?)
    ''', (timestamp, plate_text, confidence))
    print(f"[INFO] Detection added: {plate_text} at {timestamp}")

# Example usage
plate_text = "MH12AB8001"
owner_name = "harsh Kachhaway"
phone_number = "+917249246230"
confidence = 0.95

# Ensure plate info is stored before detection
add_plate(plate_text, owner_name, phone_number)
add_detection(plate_text, confidence)

# Commit and close
conn.commit()
conn.close()
