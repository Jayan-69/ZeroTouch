import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import json
import math
from enum import Enum

# Initialize PyAutoGUI with failsafe
pyautogui.FAILSAFE = False

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# MediaPipe Hand Landmark indices
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Application states
class AppState(Enum):
    MAIN = 1
    CALIBRATING = 2
    RUNNING = 3

# Calibration settings
CALIBRATION_FILE = 'hand_calibration.json'
DEFAULT_CALIBRATION = {
    'top_left': {'x': 0.1, 'y': 0.1},
    'top_right': {'x': 0.9, 'y': 0.1},
    'bottom_left': {'x': 0.1, 'y': 0.9},
    'bottom_right': {'x': 0.9, 'y': 0.9},
    'center': {'x': 0.5, 'y': 0.5}
}

# Finger landmark indices (MediaPipe Hand Landmarks)
THUMB_TIP = 4
THUMB_MCP = 2
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

# Initialize MediaPipe Hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections for drawing

# Hand connections for drawing
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Constants
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
WRIST = 0

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class HandMouseController:
    def __init__(self):
        # Initialize camera with retries
        max_retries = 3
        self.cap = None
        
        for i in range(max_retries):
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break
            print(f"Camera initialization attempt {i+1} failed, retrying...")
            time.sleep(1)
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to initialize camera. Please check your camera connection.")

        # Hand size detection
        self.hand_size = None  # Will store the size of the hand (palm width)
        self.hand_landmarks_prev = None
        
        # For smooth cursor movement
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.smoothing = 3
        
        # For click handling
        self.left_clicked = False
        
        # For scrolling
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # seconds between scrolls
        
        # Frame reduction for better hand detection
        self.frameR = 100
        
        # For frame timing
        self.last_timestamp_ms = 0
        self.latest_detection_result = None
        
        # Initialize MediaPipe Hand Landmarker
        self.initialize_hand_landmarker()
    
    def initialize_hand_landmarker(self):
        """Initialize MediaPipe Hand Landmarker."""
        base_options = mp.tasks.BaseOptions
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode
        
        # Create a hand landmarker instance with the live stream mode
        options = hand_landmarker_options(
            base_options=base_options(model_asset_path=resource_path('hand_landmarker.task')),
            running_mode=vision_running_mode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self.process_landmarker_result)
            
        self.hand_landmarker = hand_landmarker.create_from_options(options)
    
    def process_landmarker_result(self, result, output_image, timestamp_ms):
        """Callback to process the landmarker result."""
        self.latest_detection_result = result
        
    def load_calibration(self):
        """Load calibration data from file or use default."""
        try:
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return DEFAULT_CALIBRATION
        
    def save_calibration(self):
        """Save current calibration to file."""
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.calibration_points, f)
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
            
    def start_calibration(self):
        """Start the calibration process."""
        self.state = AppState.CALIBRATING
        self.current_calibration_point = 0
        self.calibration_complete = False
        print("Starting calibration...")
        
    def update_calibration(self, hand_center):
        """Update calibration with new point."""
        if self.current_calibration_point < len(self.calibration_point_names):
            point_name = self.calibration_point_names[self.current_calibration_point]
            self.calibration_points[point_name] = {'x': hand_center[0], 'y': hand_center[1]}
            self.current_calibration_point += 1
            
            if self.current_calibration_point >= len(self.calibration_point_names):
                self.calibration_complete = True
                self.save_calibration()
                self.state = AppState.RUNNING
                print("Calibration complete!")
            
            return True
        return False
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
    
    def is_finger_up(self, hand_landmarks, finger_tip, finger_pip, finger_mcp=None):
        """Check if a finger is up based on landmark positions."""
        if finger_mcp is None:
            return hand_landmarks[finger_tip].y < hand_landmarks[finger_pip].y
        else:
            # More accurate check using MCP as reference
            tip_to_mcp = hand_landmarks[finger_tip].y - hand_landmarks[finger_mcp].y
            pip_to_mcp = hand_landmarks[finger_pip].y - hand_landmarks[finger_mcp].y
            return tip_to_mcp < pip_to_mcp
    
    def count_extended_fingers(self, hand_landmarks):
        """Count how many fingers are extended (excluding thumb for accuracy)."""
        extended_count = 0
        
        # Get wrist for reference
        wrist = hand_landmarks[WRIST]
        
        # Define finger joints: (tip, pip, mcp) - EXCLUDING THUMB
        finger_joints = [
            (INDEX_FINGER_TIP, INDEX_FINGER_PIP, 5),      # Index
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP, 9),    # Middle
            (RING_FINGER_TIP, RING_FINGER_PIP, 13),       # Ring
            (PINKY_TIP, PINKY_PIP, 17)                    # Pinky
        ]
        
        # Check each finger (NOT including thumb)
        for tip_idx, pip_idx, mcp_idx in finger_joints:
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            mcp = hand_landmarks[mcp_idx]
            
            # Calculate distances
            tip_to_mcp = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
            pip_to_mcp = math.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)
            
            # Also check distance from wrist
            tip_to_wrist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            pip_to_wrist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
            
            # Finger is extended if:
            # 1. Tip is farther from MCP than PIP (finger is straightened)
            # 2. Tip is farther from wrist than PIP (finger is pointing up/out)
            if tip_to_mcp > pip_to_mcp * 1.3 and tip_to_wrist > pip_to_wrist:
                extended_count += 1
            
        return extended_count

    def detect_gesture(self, hand_landmarks, hand_type):
        """Detect hand gesture and return action."""
        if hand_type == 'Right':
            # Right hand does clicking (fist gesture)
            thumb_tip = hand_landmarks[THUMB_TIP]
            thumb_ip = hand_landmarks[THUMB_IP]
            wrist = hand_landmarks[WRIST]
            
            # Check if fingers are folded (for fist detection)
            fingers_folded = all([
                hand_landmarks[INDEX_FINGER_TIP].y > hand_landmarks[INDEX_FINGER_PIP].y,
                hand_landmarks[MIDDLE_FINGER_TIP].y > hand_landmarks[MIDDLE_FINGER_PIP].y,
                hand_landmarks[RING_FINGER_TIP].y > hand_landmarks[RING_FINGER_PIP].y,
                hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_PIP].y
            ])
            
            # Check if thumb is not extended (folded into fist)
            thumb_folded = thumb_tip.x > thumb_ip.x
            
            if fingers_folded and thumb_folded:
                return 'click'
        
        return None
    

    
    def process_hand(self, hand_landmarks, hand_type, frame):
        """Process a single hand and perform actions based on gesture."""
        h, w, _ = frame.shape
        
        # Get index finger tip coordinates
        index_x = int(hand_landmarks[INDEX_FINGER_TIP].x * w)
        index_y = int(hand_landmarks[INDEX_FINGER_TIP].y * h)
        
        # Map hand coordinates to screen coordinates
        screen_x = np.interp(index_x, (self.frameR, w - self.frameR), (0, SCREEN_WIDTH))
        screen_y = np.interp(index_y, (self.frameR, h - self.frameR), (0, SCREEN_HEIGHT))
        
        # LEFT HAND: colorful fingers for cursor movement and scrolling
        if hand_type == 'Left':
            # Count extended fingers
            extended_fingers = self.count_extended_fingers(hand_landmarks)
            
            # Show finger count for debugging
            cv2.putText(frame, f"Left Hand - Fingers: {extended_fingers}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            
            current_time = time.time()
            
            if extended_fingers == 1:
                # Single finger - move cursor
                cv2.circle(frame, (index_x, index_y), 20, GREEN, 3)
                cv2.putText(frame, "MOVE", (index_x + 25, index_y - 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
                
                # Smooth cursor movement
                self.clocX = self.plocX + (screen_x - self.plocX) / self.smoothing
                self.clocY = self.plocY + (screen_y - self.plocY) / self.smoothing
                
                # Move cursor
                pyautogui.moveTo(self.clocX, self.clocY)
                
                self.plocX, self.plocY = self.clocX, self.clocY
                
            elif extended_fingers == 2 and (current_time - self.last_scroll_time) > self.scroll_cooldown:
                # Two fingers - scroll down
                pyautogui.scroll(-40)  # Negative value scrolls down
                cv2.circle(frame, (index_x, index_y), 25, BLUE, 4)
                cv2.putText(frame, "SCROLL DOWN", (index_x - 100, index_y - 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLUE, 3)
                self.last_scroll_time = current_time
                
            elif extended_fingers == 3 and (current_time - self.last_scroll_time) > self.scroll_cooldown:
                # Three fingers - scroll up (NOTE: changed from >= 3 to == 3)
                pyautogui.scroll(40)  # Positive value scrolls up
                cv2.circle(frame, (index_x, index_y), 25, YELLOW, 4)
                cv2.putText(frame, "SCROLL UP", (index_x - 90, index_y - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3)
                self.last_scroll_time = current_time
            
            elif extended_fingers >= 4:
                # 4 or 5 fingers - do nothing, just show message
                cv2.putText(frame, "TOO MANY FINGERS", (index_x - 100, index_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                
        # RIGHT HAND: clicking with fist
        elif hand_type == 'Right':
            cv2.putText(frame, "Right Hand - Click Mode", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            
            try:
                detected_gesture = self.detect_gesture(hand_landmarks, hand_type)
                if detected_gesture == 'click':
                    if not self.left_clicked:
                        pyautogui.click()
                        self.left_clicked = True
                        # Draw red circle for click feedback
                        cv2.circle(frame, (index_x, index_y), 35, RED, 5)
                        cv2.putText(frame, "CLICK!", (index_x - 50, index_y - 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 4)
                else:
                    self.left_clicked = False
                    # Show ready to click
                    cv2.circle(frame, (index_x, index_y), 15, (255, 128, 0), 2)
                    
            except Exception as e:
                # Skip any errors in gesture detection
                pass
        
        # COLORFUL hand landmarks - different color for each finger!
        finger_colors = {
            'thumb': (255, 0, 255),      # Magenta
            'index': (0, 255, 0),        # Green
            'middle': (255, 255, 0),     # Yellow/Cyan
            'ring': (255, 0, 0),         # Blue
            'pinky': (0, 165, 255)       # Orange
        }
        
        # Define finger landmark ranges
        thumb_landmarks = [0, 1, 2, 3, 4]
        index_landmarks = [0, 5, 6, 7, 8]
        middle_landmarks = [0, 9, 10, 11, 12]
        ring_landmarks = [0, 13, 14, 15, 16]
        pinky_landmarks = [0, 17, 18, 19, 20]
        
        # Draw connections with finger-specific colors (LEFT hand only gets colorful)
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_point = (int(hand_landmarks[start_idx].x * w), 
                             int(hand_landmarks[start_idx].y * h))
                end_point = (int(hand_landmarks[end_idx].x * w), 
                           int(hand_landmarks[end_idx].y * h))
                
                # Colorful for LEFT hand, orange for RIGHT hand
                if hand_type == 'Left':
                    # Determine color based on which finger the connection belongs to
                    if start_idx in thumb_landmarks or end_idx in thumb_landmarks:
                        color = finger_colors['thumb']
                    elif start_idx in index_landmarks or end_idx in index_landmarks:
                        color = finger_colors['index']
                    elif start_idx in middle_landmarks or end_idx in middle_landmarks:
                        color = finger_colors['middle']
                    elif start_idx in ring_landmarks or end_idx in ring_landmarks:
                        color = finger_colors['ring']
                    elif start_idx in pinky_landmarks or end_idx in pinky_landmarks:
                        color = finger_colors['pinky']
                    else:
                        color = (255, 255, 255)  # White for palm
                    
                    cv2.line(frame, start_point, end_point, color, 3)
                else:
                    # Simple orange for right hand
                    cv2.line(frame, start_point, end_point, (255, 128, 0), 2)
        
        # Draw landmarks with colors
        for idx, landmark in enumerate(hand_landmarks):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            if hand_type == 'Left':
                # Colorful landmarks for left hand
                if idx in thumb_landmarks:
                    color = finger_colors['thumb']
                elif idx in index_landmarks:
                    color = finger_colors['index']
                elif idx in middle_landmarks:
                    color = finger_colors['middle']
                elif idx in ring_landmarks:
                    color = finger_colors['ring']
                elif idx in pinky_landmarks:
                    color = finger_colors['pinky']
                else:
                    color = (255, 255, 255)
                
                # Larger circles for finger tips
                if idx in [4, 8, 12, 16, 20]:  # Tips
                    cv2.circle(frame, (x, y), 12, color, -1)
                    cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 7, color, -1)
            else:
                # Simple orange for right hand
                color = (255, 128, 0)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (x, y), 10, color, -1)
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 6, color, -1)



    def run(self):
        """Run the hand mouse controller."""
        print("Starting ZeroTouch...")
        print("="*60)
        print("LEFT HAND: 1 finger = move, 2 fingers = scroll down, 3+ fingers = scroll up")
        print("RIGHT HAND: Make a fist to click")
        print("="*60)
        print("Press 'Q' to quit")
        print("\nCamera initialized successfully! Window should appear now...")
        print("If you don't see a window, check your taskbar or try Alt+Tab")

        
        frame_count = 0
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Warning: Empty camera frame.")
                    continue
                
                # Flip the frame horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to MediaPipe Image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Calculate timestamp
                timestamp = int(time.time() * 1000)
                if timestamp <= self.last_timestamp_ms:
                    timestamp = self.last_timestamp_ms + 1
                self.last_timestamp_ms = timestamp
                
                # Process the frame asynchronously
                self.hand_landmarker.detect_async(mp_image, timestamp)
                
                # Get the latest result
                detection_result = self.latest_detection_result
                
                # Process detected hands if result is available
                if detection_result and detection_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        # Get handedness (Left or Right)
                        if idx < len(detection_result.handedness):
                            hand_type = detection_result.handedness[idx][0].category_name
                            
                            # Process each hand
                            try:
                                self.process_hand(hand_landmarks, hand_type, frame)
                            except Exception as e:
                                # Skip any errors in processing
                                print(f"Error processing hand: {e}")
                
                # Show status every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    hands_detected = len(detection_result.hand_landmarks) if (detection_result and detection_result.hand_landmarks) else 0
                    print(f"Frame {frame_count}: {hands_detected} hand(s) detected")
                
                # Draw border rectangle showing the active tracking area
                h, w, _ = frame.shape
                border_color = (0, 255, 0)  # Green border
                border_thickness = 3
                
                # Draw main rectangle border
                cv2.rectangle(frame, 
                             (self.frameR, self.frameR), 
                             (w - self.frameR, h - self.frameR), 
                             border_color, border_thickness)
                
                # Draw corner markers for better visibility
                corner_length = 30
                corner_thickness = 5
                
                # Top-left corner
                cv2.line(frame, (self.frameR, self.frameR), (self.frameR + corner_length, self.frameR), border_color, corner_thickness)
                cv2.line(frame, (self.frameR, self.frameR), (self.frameR, self.frameR + corner_length), border_color, corner_thickness)
                
                # Top-right corner
                cv2.line(frame, (w - self.frameR, self.frameR), (w - self.frameR - corner_length, self.frameR), border_color, corner_thickness)
                cv2.line(frame, (w - self.frameR, self.frameR), (w - self.frameR, self.frameR + corner_length), border_color, corner_thickness)
                
                # Bottom-left corner
                cv2.line(frame, (self.frameR, h - self.frameR), (self.frameR + corner_length, h - self.frameR), border_color, corner_thickness)
                cv2.line(frame, (self.frameR, h - self.frameR), (self.frameR, h - self.frameR - corner_length), border_color, corner_thickness)
                
                # Bottom-right corner
                cv2.line(frame, (w - self.frameR, h - self.frameR), (w - self.frameR - corner_length, h - self.frameR), border_color, corner_thickness)
                cv2.line(frame, (w - self.frameR, h - self.frameR), (w - self.frameR, h - self.frameR - corner_length), border_color, corner_thickness)
                
                # Add instruction text
                cv2.putText(frame, "Keep your hand inside the GREEN BORDER", 
                           (w//2 - 280, h - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
                
                # Display the frame
                cv2.imshow('ZeroTouch', frame)
                
                # Handle key press (wait 1ms for events to be processed)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Check if window was closed (X button clicked)
                # We check this AFTER waitKey because waitKey processes the GUI events
                if cv2.getWindowProperty('ZeroTouch', cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication stopped by user.")
        finally:
            self.cleanup()

                    
    def cleanup(self):
        """Release resources."""
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hand_landmarker'):
                self.hand_landmarker.close()
            cv2.destroyAllWindows()
            print("\nHand Mouse Controller closed.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    # Create and run the controller
    controller = HandMouseController()
    controller.run()

if __name__ == "__main__":
    main()
