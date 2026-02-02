import cv2
import numpy as np
import json
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Constants
CALIBRATION_FILE = 'hand_calibration.json'

class HandCalibrator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Calibration points
        self.calibration_points = {
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None,
            'center': None
        }
        
        # Current calibration point being set
        self.current_point = 0
        self.point_names = list(self.calibration_points.keys())
        
        # Initialize MediaPipe Hand Landmarker
        base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
    def get_hand_center(self, landmarks):
        """Calculate the center of the hand using landmarks."""
        if not landmarks or not landmarks.hand_landmarks:
            return None
            
        # Use the center of the palm (landmark 0 is the wrist, 9 is the middle finger MCP)
        wrist = landmarks.hand_landmarks[0][0]
        middle_mcp = landmarks.hand_landmarks[0][9]
        
        # Calculate center between wrist and middle finger MCP
        center_x = (wrist.x + middle_mcp.x) / 2
        center_y = (wrist.y + middle_mcp.y) / 2
        
        return (center_x, center_y)
    
    def run_calibration(self):
        """Run the calibration process."""
        print("Starting hand calibration...")
        print("Please position your hand at the following points on the screen:")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Display instructions
            current_step = self.point_names[self.current_point]
            cv2.putText(frame, f"Position your hand at the {current_step.replace('_', ' ').upper()} and press SPACE",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)
            
            # Get hand center if detected
            hand_center = self.get_hand_center(detection_result)
            
            if hand_center:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(hand_center[0] * w), int(hand_center[1] * h)
                
                # Draw hand center
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Show current coordinates
                cv2.putText(frame, f"X: {x}, Y: {y}", (x + 20, y - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show current calibration points
            for i, (point_name, point) in enumerate(self.calibration_points.items()):
                if point:
                    x, y = int(point[0] * w), int(point[1] * h)
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(frame, point_name, (x + 15, y + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Show frame
            cv2.imshow('Hand Calibration', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Space to capture current point
            if key == ord(' ') and hand_center:
                self.calibration_points[current_step] = hand_center
                print(f"Saved {current_step} at {hand_center}")
                self.current_point += 1
                
                # Check if calibration is complete
                if self.current_point >= len(self.point_names):
                    if self.save_calibration():
                        print("Calibration completed and saved successfully!")
                    else:
                        print("Error saving calibration.")
                    break
            
            # 'r' to restart calibration
            elif key == ord('r'):
                self.current_point = 0
                self.calibration_points = {k: None for k in self.calibration_points}
                print("Calibration reset.")
            
            # 'q' to quit
            elif key == ord('q'):
                print("Calibration cancelled.")
                break
        
        self.cleanup()
    
    def save_calibration(self):
        """Save calibration data to a file."""
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.calibration_points, f)
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    @staticmethod
    def load_calibration():
        """Load calibration data from file."""
        try:
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return None
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = HandCalibrator()
    calibrator.run_calibration()
    
    # After calibration, you can load the data in your main application:
    # calibration_data = HandCalibrator.load_calibration()
    # if calibration_data:
    #     print("Calibration data loaded successfully!")
    #     print(calibration_data)
