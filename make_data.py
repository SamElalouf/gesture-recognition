import cv2
from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

### Functions for cropping augmented input
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

### Snap Pics
def make_data(category_names, webcam=0, size=224, landmarks_only=True):

    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    webcam = cv2.VideoCapture(webcam)
    current_pic_num = 1
    while webcam.isOpened():
            for name in category_names:
                while True:
                        try:
                            _, frame = webcam.read()
                            x, y, c = frame.shape

                            # Flip the frame vertically
                            frame = cv2.flip(frame, 1)
                            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Get hand landmark prediction
                            result = hands.process(framergb)

                            # Prepare our images, both raw and blank.
                            annotated_image = framergb.copy()
                            blank_image = np.zeros_like(annotated_image)
                            display_image = blank_image.copy()
                            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            class_name = ''

                            # post process the result
                            if result.multi_hand_landmarks:
                                landmarks = []
                                for handslms in result.multi_hand_landmarks:
                                    for lm in handslms.landmark:
                                        lmx = int(lm.x * x)
                                        lmy = int(lm.y * y)

                                        landmarks.append([lmx, lmy])

                                    # Drawing landmarks on frames
                                    mpDraw.draw_landmarks(blank_image, handslms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                                    mpDraw.draw_landmarks(display_image, handslms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                                    mpDraw.draw_landmarks(annotated_image, handslms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                                    if landmarks_only == True:
                                        cv2.putText(display_image, "Output for " + str(name) + " Category", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 's' to snap a pic.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 'n' for next category.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 'q' to quit.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.imshow("Output for " + str(name) + " Category", display_image)
                                    elif landmarks_only == False:
                                        cv2.putText(annotated_image, "Output for " + str(name) + " Category", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 's' to snap a pic.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 'n' for next category", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 'q' to quit.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.imshow("Output for " + str(name) + " Category", annotated_image)

                            else:
                                    if landmarks_only == True:
                                        cv2.putText(display_image, "Output for " + str(name) + " Category", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 's' to snap a pic.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 'n' for next category.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(display_image, "Click 'q' to quit.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.imshow("Output for " + str(name) + " Category", display_image)
                                    elif landmarks_only == False:
                                        cv2.putText(annotated_image, "Output for " + str(name) + " Category", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 's' to snap a pic.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 'n' for next category.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(annotated_image, "Click 'q' to quit.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.imshow("Output for " + str(name) + " Category", annotated_image)

                            key = cv2.waitKey(1)
                            if key == ord('s'):
                                file_name = "data/" + str(name) + "_" + str(current_pic_num) + '.png'
                                cropped = crop_image(blank_image)
                                new_image = make_square(cropped)
                                resized_image = cv2.resize(new_image, (size, size))
                                try:
                                    resized_image = cv2.resize(new_image, (size, size))
                                    cv2.imwrite(file_name, resized_image)
                                except:
                                    resized_image = resized_image
                                current_pic_num += 1
                                key = cv2.waitKey(1)


                            elif key == ord('n'):
                                cv2.destroyWindow("Output for " + str(name) + " Category")
                                break

                            elif key == ord('q'):
                                webcam.release()
                                cv2.destroyAllWindows()
                                break
                
                        except(KeyboardInterrupt):
                            print("Turning off camera.")
                            webcam.release()
                            print("Camera off.")
                            print("Program ended.")
                            cv2.destroyAllWindows()
                            break


category_names = ["C", "Palm", "Thumb", "Down", "Index","L", "Ok", "Irrelevant"]

make_data(category_names=category_names)