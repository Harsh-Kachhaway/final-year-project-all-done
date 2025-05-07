import cv2
import datetime
import threading
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import csv
import json
import os
import sys
from PIL import Image, ImageTk
import time
import subprocess
import pytesseract
from ultralytics import YOLO

print("import complit")

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

print("Loading helmet model NUMBERPLATE...")
numberplate_model = YOLO(resource_path("models/yolo11_numberplate.pt"))
print("Loading helmet model BIKE...")
bike_model = YOLO(resource_path("models/yolo11_bikedetection.pt"))
print("Loading helmet model HELMET...")
helmet_model = YOLO(resource_path("models/yolo11_helmetdetection.pt"))
print("Loading helmet model...")

def show_splash(main_func):
    splash_root = tk.Tk()
    splash_root.overrideredirect(True)

    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()

    splash_width = 358
    splash_height = 235

    position_top = int(screen_height / 2 - splash_height / 2)
    position_left = int(screen_width / 2 - splash_width / 2)

    splash_icon = Image.open(resource_path("app_icon.ico"))
    splash_icon = ImageTk.PhotoImage(splash_icon)
    splash_root.iconphoto(True, splash_icon)

    splash_root.geometry(f"{splash_width}x{splash_height}+{position_left}+{position_top}")

    image = Image.open(resource_path("splashe.png"))
    splash_image = ImageTk.PhotoImage(image)

    tk.Label(splash_root, image=splash_image).pack()

    splash_root.splash_image = splash_image

    splash_root.after(3000, lambda: [splash_root.destroy(), main_func()])
    splash_root.mainloop()

running_flags = {}
ocr_data = []
data_lock = threading.Lock()

conn = sqlite3.connect(resource_path("data/detection_data.db"), check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        plate_text TEXT,
        confidence REAL
    )
''')
conn.commit()

def save_to_db(data):
    with data_lock:
        cursor.execute("INSERT INTO detections (timestamp, plate_text, confidence) VALUES (?, ?, ?)",
                       (data['Timestamp'], data['Plate Text'], data['Confidence']))
        conn.commit()

def process_frame(frame):
    timestamp_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    number_plate_detected = False
    plate_text = ""
    bikes = []

    results_plate = numberplate_model(frame)
    for result in results_plate:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf >= 0.2:
                plate_crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                plate_text = pytesseract.image_to_string(gray, config='--psm 7').strip()

                if plate_text:
                    number_plate_detected = True
                    cv2.putText(frame, plate_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    if number_plate_detected:
        results_bike = bike_model(frame)
        for result in results_bike:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bikes.append((x1, y1, x2, y2))
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        results_helmet = helmet_model(frame)
        for result in results_helmet:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                inside_bike = any(bx1 < x1 < bx2 and by1 < y1 < by2 for bx1, by1, bx2, by2 in bikes)
                tag = "Helmet" if inside_bike else "No Helmet"
                color = (0, 255, 0) if inside_bike else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{tag} ({conf:.2f})', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if not inside_bike and plate_text:
                    data = {
                        "Timestamp": timestamp_text,
                        "Plate Text": plate_text,
                        "Helmet Status": "No Helmet",
                        "Confidence": round(conf, 2)
                    }
                    save_to_db(data)

    return frame

threads = { }
flag_lock = threading.Lock()

def camera_thread(source, stop_event):
    data_folder = "data"

    try:
        source = int(source)
        input_type = "video"
    except ValueError:
        full_path = os.path.join(data_folder, source)
        if os.path.isfile(full_path) and full_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_type = "image"
            source = full_path
        elif source.startswith("http") and not source.endswith("/video"):
            source += "/video"
            input_type = "video"
        else:
            input_type = "video"

    if input_type == "image":
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: Could not load image {source}.")
            return

        print(f"Image {source} loaded.")
        frame = process_frame(frame)
        window_name = f"Image Detection - {os.path.basename(source)}"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.imshow(window_name, frame)

        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        print(f"Image {source} display closed.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}.")
        return

    print(f"Stream {source} started.")
    window_name = f"Live Detection - {source}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if not cap.isOpened():
                print("🔌 VideoCapture not opened:", source)
            else:
                print("❌ Failed to grab frame from:", source)
            break

        frame = process_frame(frame)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"Stream {source} stopped.")

def start_detection(urls_entry, status_label):
    inputs = urls_entry.get().split(',')
    if not inputs:
        messagebox.showwarning("Input Error", "Please enter at least one camera index or URL")
        return

    status_label.config(text="Detection running...")

    for source in inputs:
        source = source.strip()
        if not source:
            continue

        if running_flags.get(source):
            continue
        running_flags[source] = True

        stop_event = threading.Event()

        t = threading.Thread(target=camera_thread, args=(source, stop_event), daemon=True)
        threads[source] = {"thread": t, "stop_event": stop_event}
        t.start()

def stop_detection(status_label):
    status_label.config(text="Stopping...")

    for source, thread_info in threads.items():
        stop_event = thread_info["stop_event"]
        stop_event.set()

    for source, thread_info in threads.items():
        t = thread_info["thread"]
        if t.is_alive():
            t.join(timeout=5)

    with flag_lock:
        threads.clear()
        running_flags.clear()

    status_label.config(text="Stopped")

def export_to_csv():
    filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
    if not filename:
        return

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Timestamp", "Plate Text", "Confidence"])
        for row in rows:
            writer.writerow(row)

def view_detections_window():
    db_window = tk.Toplevel()
    db_window.title("Detection Records")

    search_frame = tk.Frame(db_window)
    search_frame.pack(fill='x')

    search_entry = tk.Entry(search_frame)
    search_entry.pack(side='left', fill='x', expand=True, padx=5, pady=5)

    def filter_data():
        query = search_entry.get()
        for i in tree.get_children():
            tree.delete(i)
        cursor.execute("SELECT * FROM detections WHERE plate_text LIKE ? ORDER BY id DESC", (f"%{query}%",))
        for row in cursor.fetchall():
            tree.insert("", "end", values=row)

    tk.Button(search_frame, text="Search", command=filter_data).pack(side='left', padx=5)
    tk.Button(search_frame, text="Export CSV", command=export_to_csv).pack(side='right', padx=5)

    tree = ttk.Treeview(db_window, columns=("ID", "Timestamp", "Plate Text", "Confidence"), show='headings')
    tree.heading("ID", text="ID")
    tree.heading("Timestamp", text="Timestamp")
    tree.heading("Plate Text", text="Plate Text")
    tree.heading("Confidence", text="Confidence")

    scrollbar = ttk.Scrollbar(db_window, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    tree.pack(fill='both', expand=True)

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)

URLS_FILE = resource_path("data/previous_urls.json")

def load_previous_urls():
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_previous_urls(previous_urls):
    with open(URLS_FILE, 'w') as f:
        json.dump(previous_urls, f)

def add_url_row(url, scrollable_frame, previous_urls, urls_entry, status_label):
    row = tk.Frame(scrollable_frame)
    row.pack(fill='x', pady=2, padx=5)

    label = tk.Label(row, text=url, anchor='w')
    label.pack(side='left', fill='x', expand=True)

    connect_button = tk.Button(row, text="Connect", command=lambda u=url: connect_single_url(u, urls_entry, status_label))
    connect_button.pack(side='right', padx=5)

    remove_button = tk.Button(row, text="Remove", command=lambda u=url, r=row: remove_url(u, r, previous_urls))
    remove_button.pack(side='right', padx=5)

def connect_single_url(url, urls_entry, status_label):
    if not running_flags.get(url):
        urls_entry.delete(0, tk.END)
        urls_entry.insert(0, url)
        start_detection(urls_entry, status_label)

def remove_url(url, row, previous_urls):
    if url in previous_urls:
        previous_urls.remove(url)
        save_previous_urls(previous_urls)

        row.destroy()

def update_url_list(scrollable_frame, previous_urls):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    for u in previous_urls:
        add_url_row(u, scrollable_frame, previous_urls)

def start_and_store(entry_widget, status_label, previous_urls, scrollable_frame):
    input_text = entry_widget.get()
    urls = [u.strip() for u in input_text.split(',') if u.strip()]
    updated = False

    for u in urls:
        if u not in previous_urls:
            previous_urls.append(u)
            add_url_row(u, scrollable_frame, previous_urls, entry_widget, status_label)
            updated = True

    if updated:
        save_previous_urls(previous_urls)

    start_detection(entry_widget, status_label)

def connect_all_urls(previous_urls, entry_widget, status_label):
    if not previous_urls:
        messagebox.showwarning("No URLs", "No previous URLs to connect.")
        return
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, ', '.join(previous_urls))
    start_detection(entry_widget, status_label)

def main():
    subprocess.Popen(["python", resource_path("script/second.py")])

    root = tk.Tk()
    root.title("Helmet & Number Plate Detection")

    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', pady=5)

    date_label = tk.Label(top_frame, text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=('Helvetica', 10))
    date_label.pack(side='left', padx=10)

    live_label_frame = tk.Frame(top_frame)
    live_label_frame.pack(side='right', padx=10)
    red_dot = tk.Label(live_label_frame, text="●", fg="gray", font=('Helvetica', 14, 'bold'))
    red_dot.pack(side='left')
    live_status_label = tk.Label(live_label_frame, text="Idle", fg='gray', font=('Helvetica', 10, 'bold'))
    live_status_label.pack(side='left')

    tk.Label(root, text="Enter Camera Indexes or Stream URLs (comma-separated):").pack(pady=5)
    urls_entry = tk.Entry(root, width=60)
    urls_entry.pack(pady=5)

    status_label = tk.Label(root, text="Idle")
    status_label.pack(pady=5)

    buttons_frame = tk.Frame(root)
    buttons_frame.pack(pady=5)

    start_button = None
    stop_button = None

    prev_frame = tk.LabelFrame(root, text="Previously Used URLs")
    prev_frame.pack(pady=10, fill='both', padx=10)

    canvas = tk.Canvas(prev_frame, height=120)
    scrollbar = tk.Scrollbar(prev_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    previous_urls = load_previous_urls()

    for u in previous_urls:
        add_url_row(u, scrollable_frame, previous_urls, urls_entry, status_label)

    def start_and_store_wrapper():
        start_and_store(urls_entry, status_label, previous_urls, scrollable_frame)

    def connect_all_urls_wrapper():
        connect_all_urls(previous_urls, urls_entry, status_label)

    tk.Button(buttons_frame, text="Start Detection", command=start_and_store_wrapper).pack(side='left', padx=10)
    tk.Button(buttons_frame, text="Stop Detection", command=lambda: stop_detection(status_label)).pack(side='left', padx=10)
    tk.Button(root, text="Connect All Previous URLs", command=connect_all_urls_wrapper).pack(pady=5)

    view_button = tk.Button(root, text="View Detection Records", command=view_detections_window)
    view_button.pack(side='right', padx=10, pady=10, anchor='se')

    def update_labels():
        date_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if any(running_flags.values()):
            live_status_label.config(text="LIVE", fg="red")
            red_dot.config(fg="red")
        else:
            live_status_label.config(text="Idle", fg="gray")
            red_dot.config(fg="gray")
        root.after(1000, update_labels)

    update_labels()

    def on_close():
        save_previous_urls(previous_urls)
        quit_program(root, status_label)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

def quit_program(root, status_label):
    stop_detection(status_label)
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()

if __name__ == "__main__":
    show_splash(main)



