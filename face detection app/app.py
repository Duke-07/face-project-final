import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, Response

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.chdir(r"D:\face detection app")

app = Flask(__name__)
cap = cv2.VideoCapture(0)  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                analysis = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
                mood = analysis[0]['dominant_emotion']
                label = f"{gender}, {age}, {mood}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except:
                pass

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
