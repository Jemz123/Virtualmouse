import cv2
import mediapipe as mp
import pyautogui
import os
import time

# Constants for screen size and smoothing
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR = 0.5  # Adjust this to change how smooth the mouse movement is
PINCH_THRESHOLD = 0.05   # Threshold for detecting pinch gesture

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Function to open Microsoft Paint
def open_paint():
    os.system('start mspaint')
    time.sleep(3)  # Wait for Paint to open

# Function to map hand movement to screen coordinates with smoothing
def map_hand_to_screen(index_finger_tip, last_position):
    # Convert normalized coordinates to screen coordinates
    x = int(index_finger_tip.x * SCREEN_WIDTH)
    y = int(index_finger_tip.y * SCREEN_HEIGHT)

    # Smooth the mouse movement
    smooth_x = int(last_position[0] + (x - last_position[0]) * SMOOTHING_FACTOR)
    smooth_y = int(last_position[1] + (y - last_position[1]) * SMOOTHING_FACTOR)

    return smooth_x, smooth_y

# Function to handle mouse actions
def handle_mouse_actions(index_finger_tip, thumb_tip):
    # Calculate the distance for pinch gesture
    distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
    if distance < PINCH_THRESHOLD:
        pyautogui.mouseDown()  # Press down the mouse button to start drawing
    else:
        pyautogui.mouseUp()    # Release the mouse button to stop drawing

def main():
    open_paint()  # Open Paint at the start
    cap = cv2.VideoCapture(0)
    
    last_position = (0, 0)  # Initial position for mouse smoothing

    while True:
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for natural movement
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        result = hands.process(image_rgb)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get index finger and thumb tips
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Map hand position to screen coordinates
                last_position = map_hand_to_screen(index_finger_tip, last_position)
                pyautogui.moveTo(last_position[0], last_position[1])

                # Handle mouse actions for drawing
                handle_mouse_actions(index_finger_tip, thumb_tip)

                # Draw hand landmarks on the image (for debugging)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the image
        cv2.imshow('Virtual Mouse', image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
