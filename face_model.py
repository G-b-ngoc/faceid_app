import cv2
import pandas as pd
from datetime import datetime
import os
import time

prev_time = 0
# Load model đã train

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 🔥 Đúng với label khi train
import pickle

with open("labels.pickle", "rb") as f:
    label_dict = pickle.load(f)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.16.43:81/stream")
# ------------------------------------------------------------------ #

marked_people = set()
opened_file = False

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    file_name = f"attendance_{today}.xlsx"
    time_now = datetime.now().strftime("%H:%M:%S")

    # Nếu file chưa tồn tại → tạo mới
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_excel(file_name, index=False)

    # Nếu người chưa được điểm danh trong phiên chạy
    if name not in marked_people:

        df = pd.read_excel(file_name)

        # Nếu chưa có trong file hôm nay
        if name not in df["Name"].values:

            new_row = {"Name": name, "Time": time_now}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(file_name, index=False)

            print(f"✅ Đã điểm danh: {name}")

        marked_people.add(name)

# Main Chính:
frame_count = 0
marked_people = set()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Không mở được camera")
        break
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # auto focus camera:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Hiện FPS:
    # current_time = time.time()
    # fps = 1 / (current_time - prev_time)
    # prev_time = current_time

    # cv2.putText(frame, f"FPS: {int(fps)}",
    #             (20,40),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0,255,255), 2)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:

        # scale lại
        x *= 2
        y *= 2
        w *= 2
        h *= 2

        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (200,200))

        label, confidence = recognizer.predict(face_gray)

        if confidence < 65:
            name = label_dict.get(label, "Nguoi la")
            color = (0,255,0)

            mark_attendance(name)   # 🔥 thêm dòng này
        else:
            name = "Nguoi la"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{name} {int(confidence)}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
    cv2.imshow("Multi Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()