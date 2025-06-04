# ASL-Translation into Text and Speech
Sign language is one of the communication ways for the people who have hearing issues. Common people find it a challenge to communicate with deaf people, that’s where this system is useful. This project presents the development of real-time american sign language translation into text and voice. This System captures the hand gestures using a webcam  and processes using the mediapipe and openCv Python libraries. hese 3D landmark coordinates are then structured into a consistent format and passed to a trained 2D Convolutional Neural Network (CNN) for classification. The model accurately predicts ASL Gestures. The predicted output is displayed as text, with optional conversion to voice, enabling more accessible and seamless communication across hearing boundaries.

Features
1. Able to convert sign language into text in real-time.
2. Converting text-to-speech.

Model Training
1. The CNN was trained on 2D hand landmarks extracted using MediaPipe.
2. Classes: 26 letters + special tokens
3. Evaluation: ~95% accuracy on validation set

Flow Diagram

<img width="392" alt="flow" src="https://github.com/user-attachments/assets/6e9af1bb-07de-47ff-8efd-823a81b4d875" />


Results



<img width="392" alt="Capture 1" src="https://github.com/user-attachments/assets/b40c507e-85d7-45ed-b32a-7d7e251129ec" />

<img width="392" alt="Capture" src="https://github.com/user-attachments/assets/f8e5a21f-7365-4d38-8163-fe427187dc2b" />




