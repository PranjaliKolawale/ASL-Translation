import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#extract 
def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            if len(hand_landmark_list) == 21:
                return np.array(hand_landmark_list)  # Shape: (21, 3)
    return None

# -------------------------------
# Functions to Load Data
# -------------------------------
def load_train_data(folder_path, X, y):
    for folder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder)
        if os.path.isdir(subfolder_path):
            if folder.lower() == "nothing":
                label = 26
            elif folder.lower() == "space":
                label = 27
            elif folder.lower() == "delete":
                label = 28
            else:
                label = ord(folder.upper()) - ord('A')

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image {img_path}")
                        continue
                    landmarks = extract_landmarks(img)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(label)
                        print(f"✅ Processed {filename}")
                    else:
                        print(f"❌ Skipping {filename} (no landmarks detected)")

def load_test_data(folder_path, X, y):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image {img_path}")
                continue
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X.append(landmarks)
                label = ord(filename[0].upper()) - ord('A')
                if filename.lower().startswith('nothing'):
                    label = 26
                elif filename.lower().startswith('space'):
                    label = 27
                elif filename.lower().startswith('delete'):
                    label = 28
                y.append(label)
                print(f"✅ Processed {filename}")
            else:
                print(f"❌ Skipping {filename} (no landmarks detected)")

# -------------------------------
# Load and Preprocess Data
# -------------------------------
train_data_path = r"C:\Users\HP\Desktop\ASL\ASL_Alphabet_Dataset\asl_alphabet_train"
test_data_path = r"C:\Users\HP\Desktop\ASL\ASL_Alphabet_Dataset\asl_alphabet_test"

X_train, y_train = [], []
X_test, y_test = [], []

load_train_data(train_data_path, X_train, y_train)
load_test_data(test_data_path, X_test, y_test)

# Convert to numpy arrays and reshape for 2D CNN
X_train = np.array(X_train, dtype=np.float32).reshape(-1, 21, 3, 1)  # Shape: (samples, 21, 3, 1)
y_train = np.array(y_train)
X_test = np.array(X_test, dtype=np.float32).reshape(-1, 21, 3, 1)
y_test = np.array(y_test)

if len(X_train) > 0 and len(y_train) > 0:
    print(f"✅ Training samples: {len(X_train)}, Labels: {len(y_train)}")
    X_train = X_train / np.max(X_train)
    y_train = to_categorical(y_train, num_classes=29)
else:
    print("❌ No training data loaded.")

if len(X_test) > 0 and len(y_test) > 0:
    print(f"✅ Testing samples: {len(X_test)}, Labels: {len(y_test)}")
    X_test = X_test / np.max(X_test)
    y_test = to_categorical(y_test, num_classes=29)
else:
    print("❌ No testing data loaded.")

# -------------------------------
# Define, Train, and Save 2D CNN Model
# -------------------------------
if len(X_train) > 0 and len(y_train) > 0 and len(X_test) > 0 and len(y_test) > 0:
    model = Sequential([
    Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=(21, 3, 1)),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(128, (2, 2), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 1)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(29, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 Training 2D CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    print("✅ 2D CNN Model Training Complete!")

    model.save("asl_hand_landmarks_2dcnn.h5")
    print("📁 Model saved as 'asl_hand_landmarks_2dcnn.h5'")

else:
    print("❌ Error: Training or testing data not properly loaded.")
