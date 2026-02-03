import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# ================= LOGIN =================
USERNAME = "admin"
PASSWORD = "ad"

# ================= UI COLORS =================
BG = "#121212"
FG = "#E0E0E0"
CARD = "#1E1E1E"
GREEN = "#2E7D32"
BLUE = "#1565C0"
RED = "#C62828"
PURPLE = "#6A1B9A"
ORANGE = "#EF6C00"

# ================= PATHS =================
DATASET_PATH = "dataset"
CSV_FILE = "attendance.csv"
REG_MAP_FILE = "reg_map.csv"
REGISTER_IMAGES_COUNT = 25

os.makedirs(DATASET_PATH, exist_ok=True)

# ================= FACE SETUP =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ================= GLOBAL STATE =================
label_map = {}
recognized_today = {}
last_date = None

# ================= REG MAP =================
if not os.path.exists(REG_MAP_FILE):
    pd.DataFrame(columns=["Reg_No", "Name"]).to_csv(REG_MAP_FILE, index=False)

reg_df = pd.read_csv(REG_MAP_FILE, dtype=str)

# ================= SYNC CSV WITH DATASET =================
def sync_reg_map_with_dataset():
    global reg_df
    dataset_regs = {d for d in os.listdir(DATASET_PATH)
                    if os.path.isdir(os.path.join(DATASET_PATH, d))}
    csv_regs = set(reg_df["Reg_No"])

    for reg_no in dataset_regs - csv_regs:
        reg_df.loc[len(reg_df)] = [reg_no, f"User_{reg_no}"]

    reg_df.to_csv(REG_MAP_FILE, index=False)

sync_reg_map_with_dataset()

# ================= LABEL MAP =================
def load_label_map():
    label_map.clear()
    label = 0
    for reg_no in sorted(os.listdir(DATASET_PATH)):
        if os.path.isdir(os.path.join(DATASET_PATH, reg_no)):
            label_map[label] = reg_no
            label += 1

# ================= TRAIN MODEL =================
def retrain_model():
    faces, labels = [], []
    label = 0

    for reg_no in sorted(os.listdir(DATASET_PATH)):
        path = os.path.join(DATASET_PATH, reg_no)
        if not os.path.isdir(path):
            continue

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_gray = cv2.imread(img_path, 0)
            if img_gray is not None:
                faces.append(img_gray)
                labels.append(label)
        label += 1

    if faces:
        recognizer.train(faces, np.array(labels))

# ================= INITIAL TRAIN =================
load_label_map()
if os.listdir(DATASET_PATH):
    retrain_model()

# ================= ATTENDANCE CSV =================
if not os.path.exists(CSV_FILE):
    pd.DataFrame(
        columns=["Reg_No", "Name", "Date", "In_Time", "Out_Time", "Attendance"]
    ).to_csv(CSV_FILE, index=False)

df = pd.read_csv(CSV_FILE, dtype=str)

# ================= REGISTER =================
def register_student():
    global reg_df, root

    root.withdraw()
    name = simpledialog.askstring("Register", "Enter Name", parent=root)
    reg_no = simpledialog.askstring("Register", "Enter Registration Number", parent=root)

    if not name or not reg_no:
        root.deiconify()
        return

    if reg_no in reg_df["Reg_No"].values:
        messagebox.showerror("Error", "Registration already exists", parent=root)
        root.deiconify()
        return

    cap = cv2.VideoCapture(0)
    count = 0
    save_path = os.path.join(DATASET_PATH, reg_no)
    os.makedirs(save_path, exist_ok=True)

    messagebox.showinfo("Register", "Only ONE face\nPress ESC to cancel", parent=root)

    while count < REGISTER_IMAGES_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{REGISTER_IMAGES_COUNT}",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()

    if count < REGISTER_IMAGES_COUNT:
        messagebox.showwarning("Warning", "Registration incomplete", parent=root)
        return

    reg_df.loc[len(reg_df)] = [reg_no, name]
    reg_df.to_csv(REG_MAP_FILE, index=False)

    load_label_map()
    retrain_model()

    messagebox.showinfo("Success", "Student Registered", parent=root)

# ================= CAMERA =================
cap = None
running = False

def start_camera():
    global cap, running
    if running:
        return
    cap = cv2.VideoCapture(0)
    running = True
    status_label.config(text="Camera Running", fg="#4CAF50")
    update_frame()

def stop_camera():
    global running
    running = False
    status_label.config(text="Camera Stopped", fg="#FF5252")
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def update_frame():
    global df, last_date

    if not running:
        return

    today = datetime.now().date().isoformat()
    if last_date != today:
        recognized_today.clear()
        last_date = today

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (200,200))

        try:
            label_id, confidence = recognizer.predict(face_img)
        except:
            continue

        accuracy = max(0, min(100, int(100 - confidence * 1.5)))

        name = "Unknown"
        reg_no = "Unknown"

        if label_id in label_map:
            reg_no = label_map[label_id]
            row = reg_df.loc[reg_df["Reg_No"] == reg_no]

            if not row.empty:
                if confidence < 90:
                    name = row["Name"].values[0]
                    recognized_today[reg_no] = name
                else:
                    name = recognized_today.get(reg_no, "Unknown")

        now = datetime.now().strftime("%H:%M:%S")

        if name != "Unknown":
            if not ((df["Reg_No"] == reg_no) & (df["Date"] == today)).any():
                df.loc[len(df)] = [reg_no, name, today, now, now, "Present"]
            else:
                df.loc[
                    (df["Reg_No"] == reg_no) & (df["Date"] == today),
                    "Out_Time"
                ] = now

            df.to_csv(CSV_FILE, index=False)
            refresh_table()

        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{name} | {accuracy}%",
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance Camera", frame)
    root.after(10, update_frame)

# ================= EXPORT =================
def export_excel():
    today = datetime.now().date().isoformat()
    report = df[df["Date"] == today]
    if report.empty:
        messagebox.showinfo("Export", "No data today")
        return
    file = f"attendance_{today}.xlsx"
    report.to_excel(file, index=False)
    messagebox.showinfo("Export", f"Excel saved:\n{file}")

# ================= UI =================
def refresh_table():
    for row in table.get_children():
        table.delete(row)
    for _, r in df.iterrows():
        table.insert("", "end", values=list(r))

def exit_app():
    stop_camera()
    root.destroy()
def export_date_range():
    global df

    from_date = simpledialog.askstring(
        "Date Range", "Enter FROM date (YYYY-MM-DD)", parent=root
    )
    to_date = simpledialog.askstring(
        "Date Range", "Enter TO date (YYYY-MM-DD)", parent=root
    )

    if not from_date or not to_date:
        messagebox.showwarning("Warning", "Both dates are required")
        return

    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError:
        messagebox.showerror("Error", "Invalid date format\nUse YYYY-MM-DD")
        return

    if from_dt > to_dt:
        messagebox.showerror("Error", "FROM date cannot be after TO date")
        return

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    report = df[(df["Date"] >= from_dt) & (df["Date"] <= to_dt)]

    if report.empty:
        messagebox.showinfo("Report", "No attendance found in this range")
        return

    file_name = f"attendance_{from_date}_to_{to_date}.xlsx"
    report.to_excel(file_name, index=False)

    messagebox.showinfo(
        "Success",
        f"Date-range report saved:\n{file_name}"
    )

def open_main_app():
    global root, table, status_label

    root = tk.Tk()
    root.title("Face Attendance System")
    root.geometry("1200x560")
    root.configure(bg=BG)

    tk.Label(root, text="Face Attendance System",
             bg=BG, fg=FG, font=("Arial", 20, "bold")).pack(pady=10)

    btn = tk.Frame(root, bg=BG)
    btn.pack(pady=10)

    tk.Button(btn, text="Register", bg=PURPLE, fg="white", width=15,
              command=register_student).grid(row=0, column=0, padx=8)
    tk.Button(btn, text="Start Camera", bg=GREEN, fg="white", width=15,
              command=start_camera).grid(row=0, column=1, padx=8)
    tk.Button(btn, text="Stop Camera", bg=ORANGE, fg="white", width=15,
              command=stop_camera).grid(row=0, column=2, padx=8)
    tk.Button(btn, text="Export Excel", bg=BLUE, fg="white", width=15,
              command=export_excel).grid(row=0, column=3, padx=8)
    tk.Button(btn, text="Date Range Report", bg="#455A64", fg="white",
          width=18, command=export_date_range).grid(row=0, column=4, padx=8)
    tk.Button(btn, text="Exit", bg=RED, fg="white", width=15,
              command=exit_app).grid(row=0, column=5, padx=8)

    status_label = tk.Label(root, text="Camera Stopped", bg=BG, fg="#FF5252")
    status_label.pack()

    columns = ("Reg_No","Name","Date","In_Time","Out_Time","Attendance")
    table = ttk.Treeview(root, columns=columns, show="headings", height=12)
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=180, anchor="center")
    table.pack(pady=10)

    refresh_table()
    root.mainloop()

# ================= LOGIN =================
def login():
    if user_entry.get() == USERNAME and pass_entry.get() == PASSWORD:
        login_win.destroy()
        open_main_app()
    else:
        messagebox.showerror("Error", "Invalid Login")

login_win = tk.Tk()
login_win.title("Login")
login_win.geometry("320x220")
login_win.configure(bg=BG)

tk.Label(login_win, text="Admin Login",
         bg=BG, fg=FG, font=("Arial", 18, "bold")).pack(pady=15)

tk.Label(login_win, text="Username", bg=BG, fg=FG).pack()
user_entry = tk.Entry(login_win, bg=CARD, fg=FG)
user_entry.pack()

tk.Label(login_win, text="Password", bg=BG, fg=FG).pack()
pass_entry = tk.Entry(login_win, show="*", bg=CARD, fg=FG)
pass_entry.pack()

tk.Button(login_win, text="Login", bg=GREEN,
          fg="white", width=12, command=login).pack(pady=15)

login_win.mainloop()



