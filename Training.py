import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_dict = {}
current_label = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):
    label_dict[current_label] = person_name
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(current_label)

    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

print("Training xong!")

import pickle

with open("labels.pickle", "wb") as f:
    pickle.dump(label_dict, f)