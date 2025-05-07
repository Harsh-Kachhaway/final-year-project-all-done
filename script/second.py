import sqlite3
import time
from datetime import datetime
from twilio.rest import Client
      #ya wali web site pe ja ke login karna sid token aur number milanga https://www.twilio.com/console
TWILIO_ACCOUNT_SID = 'Yah pe sid dalna'
TWILIO_AUTH_TOKEN = 'yah pe auth token'
TWILIO_PHONE_NUMBER = 'or yaha pe twilio phone number'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_message(phone_number, owner_name, plate_text):
    try:
        message = client.messages.create(
            body=f"Hello {owner_name}, your vehicle with plate {plate_text} was detected.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"[MESSAGE SENT] To: {phone_number} | SID: {message.sid}")
    except Exception as e:
        print(f"[ERROR] Could not send message to {phone_number}: {e}")

def ensure_tables_exist(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            plate_text TEXT,
            confidence REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            owner_name TEXT,
            phone_number TEXT,
            message_sent BOOLEAN DEFAULT 0
        )
    ''')

def reset_message_flags_if_new_day(cursor, current_date):
    cursor.execute("SELECT value FROM settings WHERE key = 'last_reset_date'")
    row = cursor.fetchone()

    if row is None or row[0] != current_date:
        cursor.execute("UPDATE plates SET message_sent = 0")
        cursor.execute("DELETE FROM detections")
        print("[INFO] Reset message_sent flags and cleared detections for a new day.")

        if row is None:
            cursor.execute("INSERT INTO settings (key, value) VALUES ('last_reset_date', ?)", (current_date,))
        else:
            cursor.execute("UPDATE settings SET value = ? WHERE key = 'last_reset_date'", (current_date,))
        return True
    return False

def check_matching_plates_and_send_messages():
    conn = sqlite3.connect('data/detection_data.db')
    cursor = conn.cursor()

    ensure_tables_exist(cursor)
    conn.commit()

    today = datetime.now().strftime('%Y-%m-%d')
    reset_message_flags_if_new_day(cursor, today)

    cursor.execute('''
        SELECT DISTINCT d.plate_text, p.phone_number, p.owner_name
        FROM detections d
        JOIN plates p ON d.plate_text = p.plate_text
        WHERE p.message_sent = 0
    ''')
    results = cursor.fetchall()

    if results:
        print("[INFO] Sending messages to matched owners:")
        for plate, phone, owner in results:
            send_message(phone, owner, plate)
            cursor.execute("UPDATE plates SET message_sent = 1 WHERE plate_text = ?", (plate,))
    else:
        print("[INFO] No unmatched plates found or messages already sent.")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    last_date = datetime.now().strftime('%Y-%m-%d')
    while True:
        now = datetime.now()
        current_date = now.strftime('%Y-%m-%d')

        if current_date != last_date:
            print(f"\n[INFO] Date changed to {current_date}. Resetting flags...")
            last_date = current_date

        print(f"\n[RUN] Checking at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        check_matching_plates_and_send_messages()
        time.sleep(60)
