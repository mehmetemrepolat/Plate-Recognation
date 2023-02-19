import numpy as np
import cv2
import pytesseract
from PIL import Image
import socket
from flask import Flask, render_template, Response

app = Flask("__main__")

camera = cv2.VideoCapture(0)
def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = gray
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":

    app.run(debug=True)