from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from flask import request, redirect, url_for, flash


app = Flask(__name__)

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imwrite('static/result.png', img)

        return redirect(url_for('main'))

@app.route('/hand_and_face_detection')
def hand_and_face_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open video capture.")
        return "Error: Failed to open video capture."

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def detect_objects():
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from video capture.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Face Detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Hand Detection
            results = hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Face Mesh
            results = faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                    mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)