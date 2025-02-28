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
cooldown = 0.4  # seconds - decreased from 0.8
last_action_time = time.time()
block_update_interval = 0.05  # Time between continuous block updates - decreased from 0.1

# Track position history for movement detection
position_history = []
max_history = 4  # Reduced to decrease detection lag

# State tracking
head_state = "center"  # Can be "left", "center", "right"
last_gesture = None    # Track the last detected gesture to prevent repeats
gesture_locked = False # Lock to prevent repeated detections
block_active = False   # Track if block is currently active
last_block_time = 0    # Last time block was refreshed

# Uppercut detection parameters
uppercut_threshold = -0.08  # Negative value for upward movement
uppercut_cooldown = 0.3  # Decreased from 0.5 for faster responses
last_uppercut_time = 0

# Upper punch zone parameters
upper_punch_cooldown = 0.3  # Decreased from 0.5 for faster responses
last_upper_punch_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Get frame dimensions for drawing
    height, width, _ = frame.shape

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
        
        # Check if cooldown has elapsed and we should unlock gesture detection for non-block gestures
        if current_time - last_action_time > cooldown and gesture_locked and last_gesture != "block":
            gesture_locked = False
            last_gesture = None
        
        # Define the y-position range for blocking (around lip height with some tolerance)
        lip_y_position = nose.y + 0.05  # Approximate lip height based on nose position
        wrist_y_tolerance = 0.08  # Tolerance for wrist height
        
        # Check if both wrists are in the blocking position
        is_blocking_position = (
            abs(left_wrist.y - lip_y_position) < wrist_y_tolerance and 
            abs(right_wrist.y - lip_y_position) < wrist_y_tolerance and
            0.35 < left_wrist.x < 0.65 and 
            0.35 < right_wrist.x < 0.65
        )
        
        # Handle block state with press-and-hold functionality
        if is_blocking_position:
            if not block_active:
                # First time entering block position - press the key
                print("Block Started - Key Down (E)")
                pyautogui.keyDown('e')  # Press and hold the key
                last_block_time = current_time
                block_active = True
                last_gesture = "block"
                # Only lock gestures for non-block actions
                gesture_locked = True
        else:
            # When no longer blocking, immediately release the block
            if block_active:
                print("Block Released - Key Up (E)")
                pyautogui.keyUp('e')  # Release the key
                block_active = False
                last_action_time = current_time
        
        # Only process other movements if we have enough history and block isn't active
        if len(position_history) >= 3 and not block_active:  # Removed the gesture_locked check for movement
            # Get movement vectors - only use recent frames (last 3) to detect quicker movements
            nose_movement = position_history[-1]["nose"] - position_history[-3]["nose"]
            left_wrist_movement = position_history[-1]["left_wrist"] - position_history[-3]["left_wrist"]
            right_wrist_movement = position_history[-1]["right_wrist"] - position_history[-3]["right_wrist"]
            
            # Debug info - print nose position and movement
            print(f"Nose position: {nose.x:.2f}, Movement: {nose_movement[0]:.4f}")
            
            # 1. HEAD MOVEMENT DETECTION (center to left = left, center to right = right)
            # Detect significant horizontal head movement
            if abs(nose_movement[0]) > 0.05:  # X-axis movement threshold
                current_time = time.time()
                
                # Only process movements if cooldown has elapsed
                if current_time - last_action_time > cooldown or last_gesture != "left" and last_gesture != "right":
                    # Moved from center to left (negative x movement)
                    if nose_movement[0] < -0.05 and nose.x < 0.60:  # Increased movement threshold
                        print("Move Left (S)")
                        pyautogui.press('s')
                        last_action_time = current_time
                        last_gesture = "left"
                        position_history = position_history[-2:]  # Keep only recent history
                    
                    # Moved from center to right (positive x movement)
                    elif nose_movement[0] > 0.05 and nose.x > 0.40:  # Increased movement threshold
                        print("Move Right (D)")
                        pyautogui.press('d')
                        last_action_time = current_time
                        last_gesture = "right"
                        position_history = position_history[-2:]  # Keep only recent history
            
            # 2. UPPERCUT DETECTION
            # Detect significant upward movement of both wrists
            if (current_time - last_uppercut_time) > uppercut_cooldown:
                # Check if both wrists moved upward significantly
                if (left_wrist_movement[1] < uppercut_threshold and 
                    right_wrist_movement[1] < uppercut_threshold and
                    left_wrist.y < left_shoulder.y and 
                    right_wrist.y < right_shoulder.y):
                    
                    print(f"Uppercut detected! Left: {left_wrist_movement[1]:.4f}, Right: {right_wrist_movement[1]:.4f}")
                    print("Uppercut (K)")
                    pyautogui.press('k')
                    last_uppercut_time = current_time
                    last_action_time = current_time  # Also update general cooldown
                    last_gesture = "uppercut"
                    position_history = position_history[-2:]  # Keep only recent history
                    
            # 3. UPPER PUNCH DETECTION USING TARGET ZONE
            # Define upper punch zone above the head
            upper_punch_zone_y_top = nose.y - 0.25  # Zone starts above the nose
            upper_punch_zone_y_bottom = nose.y - 0.05  # Zone ends closer to the nose
            upper_punch_zone_x_left = nose.x - 0.15  # Zone extends left of nose
            upper_punch_zone_x_right = nose.x + 0.15  # Zone extends right of nose
            
            # Draw the upper punch target zone
            upper_zone_left = int(width * upper_punch_zone_x_left)
            upper_zone_right = int(width * upper_punch_zone_x_right)
            upper_zone_top = int(height * upper_punch_zone_y_top)
            upper_zone_bottom = int(height * upper_punch_zone_y_bottom)
            
            # Determine zone color based on cooldown status
            zone_color = (0, 255, 0) if (current_time - last_upper_punch_time) > upper_punch_cooldown else (0, 0, 255)
            cv2.rectangle(frame, 
                        (upper_zone_left, upper_zone_top), 
                        (upper_zone_right, upper_zone_bottom), 
                        zone_color, 2)
            
            # Check if either wrist is in the upper punch zone and cooldown elapsed
            if (current_time - last_upper_punch_time) > upper_punch_cooldown:
                left_wrist_in_zone = (
                    upper_punch_zone_x_left < left_wrist.x < upper_punch_zone_x_right and
                    upper_punch_zone_y_top < left_wrist.y < upper_punch_zone_y_bottom
                )
                
                right_wrist_in_zone = (
                    upper_punch_zone_x_left < right_wrist.x < upper_punch_zone_x_right and
                    upper_punch_zone_y_top < right_wrist.y < upper_punch_zone_y_bottom
                )
                
                if left_wrist_in_zone or right_wrist_in_zone:
                    wrist_name = "Left" if left_wrist_in_zone else "Right"
                    print(f"{wrist_name} wrist in upper punch zone!")
                    print("Upper Punch (K)")
                    pyautogui.press('k')
                    last_upper_punch_time = current_time
                    last_action_time = current_time  # Also update general cooldown
                    last_gesture = "upper_punch"
                    position_history = position_history[-2:]  # Keep only recent history

        # Display the webcam feed with landmarks drawn
        cv2.putText(frame, f"Nose X: {nose.x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if len(position_history) >= 3:
            cv2.putText(frame, f"Movement: {nose_movement[0]:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add uppercut movement debug info
            cv2.putText(frame, f"L Wrist Y: {left_wrist_movement[1]:.4f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"R Wrist Y: {right_wrist_movement[1]:.4f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display block status
        if block_active:
            cv2.putText(frame, "BLOCKING", (width//2-60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display uppercut/upper punch status if recently performed
        if current_time - last_uppercut_time < 0.5:  # Show for half a second
            cv2.putText(frame, "UPPERCUT", (width//2-60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        elif current_time - last_upper_punch_time < 0.5:  # Show for half a second
            cv2.putText(frame, "UPPER PUNCH", (width//2-80, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Highlight block detection zone
        block_left = int(width * 0.35)
        block_right = int(width * 0.65)
        block_y = int(height * lip_y_position)
        block_y_tolerance = int(height * wrist_y_tolerance)
        # Highlight the block zone with a color based on whether blocking is active
        block_color = (0, 0, 255) if block_active else (0, 255, 255)
        cv2.rectangle(frame, 
                    (block_left, block_y - block_y_tolerance), 
                    (block_right, block_y + block_y_tolerance), 
                    block_color, 2)

    # Draw coordinate grid for fine-tuning (even if no landmarks detected)
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
    left_thresh = int(width * 0.40) 
    right_thresh = int(width * 0.60)
    cv2.line(frame, (left_thresh, 0), (left_thresh, height), (0, 0, 255), 2)
    cv2.line(frame, (right_thresh, 0), (right_thresh, height), (0, 0, 255), 2)
    
    cv2.imshow('Head Movement Control - Press Q to Quit', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ensure all keys are released before closing
if block_active:
    pyautogui.keyUp('e')

# Cleanup
cap.release()
cv2.destroyAllWindows()