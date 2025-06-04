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
sentence = []  # List to store completed words
current_word = ""  # Current word being spelled
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
# Text-to-Speech Helper
# -------------------------------
def speak_text(text):
    if not text.strip():
        return
    
    clean_text = ' '.join(text.split())
    tts = gTTS(text=clean_text, lang='en')
    filename = f"temp_{random.randint(1000,9999)}.mp3"
    tts.save(filename)
    
    if os.name == "nt":
        os.system(f"start {filename}")
    elif os.name == "posix":
        os.system(f"afplay {filename} &")
    else:
        os.system(f"mpg123 {filename} &")

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
    return render_template('index.html', 
                         sentence=' '.join(sentence), 
                         current_word=current_word)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify(sentence=' '.join(sentence), 
                 current_word=current_word)

@app.route('/current_word')
def get_current_word():
    return jsonify(current_word=current_word)

@app.route('/clear', methods=['POST'])
def clear_sentence():
    global sentence, current_word
    sentence = []
    current_word = ""
    return jsonify(message="cleared")

@app.route('/speak', methods=['POST'])
def speak_sentence():
    global sentence, current_word
    # Combine completed words and current word (if any)
    full_text = ' '.join(sentence + [current_word]) if current_word else ' '.join(sentence)
    speak_text(full_text)
    return jsonify(message="spoken")

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('query', '')
    return jsonify(suggestions=get_autocomplete_suggestions(query))

# -------------------------------
# Frame Generator (Updated)
# -------------------------------
def generate_frames():
    global sentence, current_word, last_prediction_time
    
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
                        if current_word:
                            sentence.append(current_word)
                            current_word = ""
                    elif current_char == "Delete":
                        if current_word:
                            current_word = current_word[:-1]
                        elif sentence:
                            sentence.pop()
                    elif current_char != "Nothing":
                        current_word += current_char.lower()  # Use lowercase for natural words
                    last_prediction_time = time.time()

        # Display information
        cv2.putText(frame, f"Predicted: {current_char}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Current Word: {current_word}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Sentence: {' '.join(sentence)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------------
# Main Entry
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)