# command ในนี้ อิงตาม flowchart นะครับ
#import library
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load model from folder
model = load_model('models/imageclassifier.h5')

# open webcam
cap = cv2.VideoCapture(0)  # ใช้ 0 สำหรับกล้องหลักของเครื่อง

# if webcam open?
if not cap.isOpened():
    # print cannot open camera
    print("Cannot open webcam")
    exit()

# While loop = True
while True:
    # Read frame from cam
    ret, frame = cap.read()

    # if false to read
    if not ret:
        # print failed to grab frame
        print("Failed to grab frame")
        break

    # Convert frame BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image to 256x256 
    resize = tf.image.resize(img, (256, 256))
    # Normalize and expand dims
    input_img = np.expand_dims(resize / 255.0, axis=0)  

    # Predict using model
    yhat = model.predict(input_img)

    # Interpret prediction
    # if > 0.5 = sad
    # else = happy
    label = "Sad" if yhat > 0.5 else "Happy"
    color = (0, 0, 255) if label == "Sad" else (0, 255, 0)

    cv2.putText(frame, f'Prediction: {label}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show image in window        
    cv2.imshow('Webcam Feed', frame)

    # if key press == 'q' ?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam & close GUI
cap.release()
cv2.destroyAllWindows()
