import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam (device index 0)
cap = cv2.VideoCapture(0)

# Cooldown time to prevent multiple key presses from one gesture
cooldown = 0.8  # seconds - increased from 0.5
last_action_time = time.time()

# Track position history for movement detection
position_history = []
max_history = 5  # Reduced from 10 to decrease detection lag

# State tracking
head_state = "center"  # Can be "left", "center", "right"
last_gesture = None    # Track the last detected gesture to prevent repeats
gesture_locked = False # Lock to prevent repeated detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Draw pose landmarks on the frame for visualization
    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get landmark coordinates
        landmarks = result.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Store position history for movement detection
        position_history.append({
            "nose": np.array([nose.x, nose.y, nose.z]),
            "left_wrist": np.array([left_wrist.x, left_wrist.y, left_wrist.z]),
            "right_wrist": np.array([right_wrist.x, right_wrist.y, right_wrist.z]),
            "left_elbow": np.array([left_elbow.x, left_elbow.y, left_elbow.z]),
            "right_elbow": np.array([right_elbow.x, right_elbow.y, right_elbow.z]),
            "left_shoulder": np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z]),
            "right_shoulder": np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        })
        
        if len(position_history) > max_history:
            position_history.pop(0)

        current_time = time.time()
        
        # Check if cooldown has elapsed and we should unlock gesture detection
        if current_time - last_action_time > cooldown and gesture_locked:
            gesture_locked = False
            last_gesture = None
        
        # Only process movements if we have enough history and gestures aren't locked
        if len(position_history) >= 3 and not gesture_locked:
            # Get movement vectors - only use recent frames (last 3) to detect quicker movements
            nose_movement = position_history[-1]["nose"] - position_history[-3]["nose"]
            
            # Debug info - print nose position and movement
            print(f"Nose position: {nose.x:.2f}, Movement: {nose_movement[0]:.4f}")
            
            # 1. HEAD MOVEMENT DETECTION (center to left = left, center to right = right)
            # Detect significant horizontal head movement
            if abs(nose_movement[0]) > 0.05:  # X-axis movement threshold
                # Moved from center to left (negative x movement)
                if nose_movement[0] < 0 and nose.x < 0.60 and last_gesture != "left":
                    print("Move Left (S)")
                    pyautogui.press('s')
                    last_action_time = current_time
                    last_gesture = "left"
                    gesture_locked = True
                    position_history = position_history[-2:]  # Keep only recent history
                
                # Moved from center to right (positive x movement)
                elif nose_movement[0] > 0 and nose.x > 0.40 and last_gesture != "right":
                    print("Move Right (D)")
                    pyautogui.press('d')
                    last_action_time = current_time
                    last_gesture = "right"
                    gesture_locked = True
                    position_history = position_history[-2:]  # Keep only recent history

    # Display the webcam feed with landmarks drawn
    cv2.putText(frame, f"Nose X: {nose.x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if len(position_history) >= 3:
        cv2.putText(frame, f"Movement: {nose_movement[0]:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw coordinate grid for fine-tuning
    height, width, _ = frame.shape
    
    # Draw horizontal lines at 0.1 increments
    for i in range(1, 10):
        y_pos = int(height * (i / 10))
        cv2.line(frame, (0, y_pos), (width, y_pos), (50, 50, 50), 1)
        # Add y-coordinate labels
        cv2.putText(frame, f"{i/10:.1f}", (5, y_pos - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw vertical lines at 0.1 increments
    for i in range(1, 10):
        x_pos = int(width * (i / 10))
        cv2.line(frame, (x_pos, 0), (x_pos, height), (50, 50, 50), 1)
        # Add x-coordinate labels
        cv2.putText(frame, f"{i/10:.1f}", (x_pos + 5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Highlight the current thresholds used in your code
    # For head movement
    left_thresh = int(width * 0.40)  # Changed from 0.45
    right_thresh = int(width * 0.60)  # Changed from 0.55
    cv2.line(frame, (left_thresh, 0), (left_thresh, height), (0, 0, 255), 2)
    cv2.line(frame, (right_thresh, 0), (right_thresh, height), (0, 0, 255), 2)
    
    cv2.imshow('Head Movement Control - Press Q to Quit', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
