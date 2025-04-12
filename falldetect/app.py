from flask import Flask, render_template, jsonify
import threading
import time
import cv2
from ollama import chat

app = Flask(__name__)

# Constants
IMAGE_PATH = 'static/webcam_snapshot.jpg'
MODEL_NAME = 'llama3.2-vision'

# Global control flag
running = False
last_result = "No result yet"

def capture_image(path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    # Allow camera to warm up and auto-adjust
    time.sleep(2)

    # Grab a few frames to let auto-exposure settle
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(path, frame)
    else:
        raise RuntimeError("Failed to capture image")

    cap.release()

def describe_image(path):
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'user',
                'content': (
                    "Carefully examine the image and determine if it shows a person who has fallen on the ground, "
                    "not sitting or lying intentionally. Reply strictly with 'yes' if you detect a fall. "
                    "Reply 'no' if there is no fallen person. Do not include anything else in your answer."
                ),
                'images': [path],
            }
        ]
    )
    return response.message.content.strip().lower()

def detection_loop():
    global running, last_result
    while running:
        try:
            capture_image(IMAGE_PATH)
            last_result = describe_image(IMAGE_PATH)
        except Exception as e:
            last_result = f"Error: {e}"
        time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=detection_loop)
        thread.start()
    return jsonify(status="started")

@app.route('/stop')
def stop_detection():
    global running
    running = False
    return jsonify(status="stopped")

@app.route('/result')
def get_result():
    global last_result
    is_alert = "yes" in last_result  # very basic check
    return jsonify(result=last_result, alert=is_alert)


if __name__ == '__main__':
    app.run(debug=True)
