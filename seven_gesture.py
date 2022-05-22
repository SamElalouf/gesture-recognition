import cv2
import os
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import urllib.request
from tensorflow.keras.models import load_model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Functions for cropping augmented input

def make_square(img):
    big_dim = np.argmax(img.shape[:2])
    small_dim = 1 - big_dim

    size = img.shape[big_dim]
    new_img = np.zeros((size, size, 3))
    offset = int((img.shape[big_dim] - img.shape[small_dim]) / 2)
    if small_dim == 0:  
        new_img[offset: offset + img.shape[small_dim], :] = img
    else:
        new_img[:, offset: offset + img.shape[small_dim]] = img
    return new_img
def crop_image(img):
    # img is 2D image data
    mask = img > 0
    mask = mask.any(2)
    return img[np.ix_(mask.any(1), mask.any(0))]

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load model
url = "https://www.dropbox.com/s/vq8c9yu4or0wkc0/06-0.02.hdf5?dl=1"
local_file, headers = urllib.request.urlretrieve(url, "models/gesture_model.hdf5")
model = load_model(local_file)

class_names = ["C" , "Down", "Index" , "Irrelevant", "L" , "Ok" , "Palm" , "Thumb"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize last gesture variable
last_gesture = None
consec_gesture_count = 0

while True:
    # Read each frame from the webcam

    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Binary Hasing variables
    annotated_image = framergb.copy()
    blank_image = np.zeros_like(annotated_image)

    # print(result)
    
    class_name = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(blank_image, handslms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

            cropped = crop_image(blank_image)
            new_image = make_square(cropped)
            resized_image = new_image

            resized_image = cv2.resize(new_image, (224, 224))
            try:
                resized_image = cv2.resize(new_image, (224, 224))
                cv2.imwrite("resize.png", resized_image)
            except:
                resized_image = resized_image





            # Predict gesture
            prediction = model.predict(np.array([resized_image]))
            # print(prediction)
            class_name = class_names[np.argmax(prediction)]
            

            if class_name == last_gesture:
                consec_gesture_count += 1
            else:
                consec_gesture_count = 0

            print(consec_gesture_count, class_name)
            if consec_gesture_count == 6:
                if class_name == "C" :
                    os.system("spotify play")
                elif class_name == "Palm" :
                    os.system("spotify pause")

                else:
                    if class_name == "Thumb":
                        os.system("spotify volume up 20")
                    elif class_name == "Down":
                        os.system("spotify volume down 20")
                    elif class_name == "Index":
                        os.system("spotify next")
                    elif class_name == "L":
                        os.system("spotify previous")
                    elif class_name == "Ok":
                        os.system("spotify history")
                    time.sleep(1)
                    consec_gesture_count = 0

            last_gesture = class_name
    text = "Palm = Play, C = Pause, Thumb Up = Vol Up, Thumb Down = Vol Down, Point up = Next, L = Last"
    # show the prediction on the frame
    #cv2.putText(blank_image, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                 #  1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", blank_image) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()