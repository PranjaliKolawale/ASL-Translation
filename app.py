import os
import cv2
import time
import random
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from gtts import gTTS
from nltk.corpus import words
from nltk import download
from rapidfuzz import process
from collections import deque, Counter
import tensorflow as tf
import mediapipe as mp

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
model = tf.keras.models.load_model("model/asl_hand_landmarks_2dcnn.h5")
cap = cv2.VideoCapture(0)

# -------------------------------
# Global Variables
# -------------------------------
labels = [chr(i) for i in range(65, 91)] + ["Nothing", "Space", "Delete"]
sentence = ""
current_word = ""
prediction_buffer = deque(maxlen=10)
last_prediction_time = time.time()
cooldown_seconds = 1.5

# -------------------------------
# NLP Setup
# -------------------------------
try:
    words.words()
except:
    download('words')
word_list = words.words()

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------------
# Extract Landmarks
# -------------------------------
def extract_landmarks_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm = [[pt.x, pt.y, pt.z] for pt in hand.landmark]
            if len(lm) == 21:
                return np.array(lm).reshape(1, 21, 3, 1).astype(np.float32)
    return None

# -------------------------------
# Autocomplete Helper
# -------------------------------
def get_autocomplete_suggestions(query):
    results = process.extract(query, word_list, limit=5)
    return [r[0] for r in results]

# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    if request.form['username'] == 'testuser' and request.form['password'] == 'pass@123':
        return redirect(url_for('index'))
    return "Invalid credentials, please try again."

@app.route('/index')
def index():
    return render_template('index.html', sentence=sentence)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify(sentence=sentence)

@app.route('/current_word')
def get_current_word():
    return jsonify(current_word=current_word)

@app.route('/clear', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ""
    return jsonify(message="cleared")

# @app.route('/speak', methods=['POST'])
# def speak_sentence():
#     global sentence
#     tts = gTTS(text=sentence, lang='en')
#     filename = f"temp_{random.randint(1000,9999)}.mp3"
#     tts.save(filename)
#     os.system(f"start {filename}" if os.name == "nt" else f"afplay {filename}")
#     return jsonify(message="spoken")
@app.route('/speak', methods=['POST'])
def speak_sentence():
    global sentence
    # Clean up multiple spaces or unintended formatting
    clean_sentence = ' '.join(sentence.split())
    if not clean_sentence.strip():
        return jsonify(message="Nothing to speak.")
    
    tts = gTTS(text=clean_sentence, lang='en')
    filename = f"temp_{random.randint(1000,9999)}.mp3"
    tts.save(filename)
    
    # Use platform-specific audio playback
    if os.name == "nt":
        os.system(f"start {filename}")  # Windows
    elif os.name == "posix":
        os.system(f"afplay {filename} &")  # macOS
    else:
        os.system(f"mpg123 {filename} &")  # Linux alternative
    
    return jsonify(message="spoken")


@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('query', '')
    return jsonify(suggestions=get_autocomplete_suggestions(query))

# -------------------------------
# Frame Generator
# -------------------------------
def generate_frames():
    global sentence, last_prediction_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_char = "..."

        landmarks = extract_landmarks_from_frame(frame)
        if landmarks is not None:
            prediction = model.predict(landmarks)
            class_idx = np.argmax(prediction)
            current_char = labels[class_idx]
            prediction_buffer.append(current_char)

            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common = Counter(prediction_buffer).most_common(1)[0]
                if most_common[1] > 7 and (time.time() - last_prediction_time) > cooldown_seconds:
                    if current_char == "Space":
                        sentence += " "
                    elif current_char == "Delete":
                        sentence = sentence[:-1]
                    elif current_char != "Nothing":
                        sentence += current_char
                    last_prediction_time = time.time()

        # Display predicted character
        cv2.putText(frame, f"Predicted: {current_char}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------------
# Main Entry
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

