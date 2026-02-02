import cv2
import pyautogui
import numpy as np
import time
import math

# Disable PyAutoGUI's fail-safe
pyautogui.FAILSAFE = False

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Smoothing factor for mouse movement
SMOOTHING = 5

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

class HandMouseController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)
        self.frameR = 100  # Frame reduction for better hand detection
        self.smoothening = SMOOTHING
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.last_click_time = 0
        self.last_volume_change = time.time()
        self.media_action = MediaAction.NONE
        self.media_action_time = 0
        self.cursor_radius = 15
        self.cursor_growing = True
        
        # Store hand states
        self.left_hand = {
            'gesture': Gesture.NONE,
            'position': (0, 0),
            'last_gesture': Gesture.NONE,
            'gesture_start_time': 0
        }
        
        self.right_hand = {
            'gesture': Gesture.NONE,
            'position': (0, 0),
            'last_gesture': Gesture.NONE,
            'gesture_start_time': 0
        }

    def get_hand_center(self, contour):
        """Get the center of the hand contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        return None

    def detect_hand(self, img):
        """Detect hand using background subtraction."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(img)
        
        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour (hand)
            hand_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area to avoid small noise
            if cv2.contourArea(hand_contour) < 1000:  # Minimum area threshold
                return None
                
            # Get hand center
            hand_center = self.get_hand_center(hand_contour)
            if hand_center:
                # Draw contour and center
                cv2.drawContours(img, [hand_contour], 0, (0, 255, 0), 2)
                cv2.circle(img, hand_center, 7, (255, 255, 255), -1)
                return hand_center
        return None
    
    def run(self):
        """Main loop for hand tracking and mouse control."""
        print("Starting Hand Mouse Controller...")
        print("Move your hand to control the mouse")
        print("Move to bottom of the screen to click")
        print("Press 'q' to quit")
        
        # Let the background subtractor learn the background
        print("Learning background... Please wait...")
        for _ in range(60):
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture background")
                return
            cv2.waitKey(30)
        print("Background learning complete!")
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from camera")
                break
                
            img = cv2.flip(img, 1)  # Mirror the image
            
            # Detect hand
            hand_pos = self.detect_hand(img)
            
            if hand_pos:
                # Map hand position to screen coordinates
                screen_x = np.interp(hand_pos[0], (self.frameR, CAMERA_WIDTH - self.frameR), (0, SCREEN_WIDTH))
                screen_y = np.interp(hand_pos[1], (self.frameR, CAMERA_HEIGHT - self.frameR), (0, SCREEN_HEIGHT))
                
                # Apply smoothing
                self.clocX = self.plocX + (screen_x - self.plocX) / self.smoothening
                self.clocY = self.plocY + (screen_y - self.plocY) / self.smoothening
                
                # Move the mouse
                pyautogui.moveTo(self.clocX, self.clocY)
                self.plocX, self.plocY = self.clocX, self.clocY
                
                # Draw cursor
                cv2.circle(img, hand_pos, 10, self.cursor_color, -1)
                
                # Check for click (hand close to bottom of the frame)
                if hand_pos[1] > CAMERA_HEIGHT - 100:  # If hand is in the bottom 100 pixels
                    if not self.is_clicking:  # Just started clicking
                        self.click_start_time = time.time()
                        self.is_clicking = True
                        self.cursor_color = RED
                    
                    # If hand stays in click zone for 0.3 seconds, perform click
                    if time.time() - self.click_start_time > 0.3:
                        pyautogui.click()
                        self.click_start_time = time.time()  # Reset timer
                else:
                    self.is_clicking = False
                    self.cursor_color = GREEN
            
            # Draw click zone
            cv2.rectangle(img, (0, CAMERA_HEIGHT - 100), (CAMERA_WIDTH, CAMERA_HEIGHT), (0, 0, 255), 2)
            cv2.putText(img, "Click Zone", (10, CAMERA_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the image
            cv2.imshow("Hand Mouse Controller", img)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        

    
    def get_hand_center(self, contour):
        """Get the center of a hand contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        return (0, 0)
    

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return Gesture.NONE
            
        solidity = float(area) / hull_area
        
        if solidity > 0.8:  # Fist
            return Gesture.FIST
        else:  # Open hand
            return Gesture.OPEN


                
    def handle_media_controls(self, gesture):
        """Handle media controls based on left hand gestures."""
        current_time = time.time()
        
        # Only process media controls if enough time has passed since the last action
        if current_time - self.last_volume_change < VOLUME_CHANGE_DELAY:
            return
            
        if gesture == Gesture.THUMBS_UP:
            keyboard.press_and_release('volume up')
            self.media_action = MediaAction.VOLUME_UP
            self.last_volume_change = current_time
            self.media_action_time = current_time
        elif gesture == Gesture.THUMBS_DOWN:
            keyboard.press_and_release('volume down')
            self.media_action = MediaAction.VOLUME_DOWN
            self.last_volume_change = current_time
            self.media_action_time = current_time
        elif gesture == Gesture.VICTORY:
            keyboard.press_and_release('play/pause media')
            self.media_action = MediaAction.PLAY_PAUSE
            self.last_volume_change = current_time
            self.media_action_time = current_time
            
    def draw_ui(self, img):
        """Draw the user interface elements."""
        # Draw instructions
        cv2.putText(img, 'Right Hand: Move Mouse', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, 'Left Hand Fist: Click', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, 'Left Hand Thumbs Up: Volume Up', (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, 'Left Hand Thumbs Down: Volume Down', (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, 'Left Hand Victory: Play/Pause', (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        # Draw current gesture and action
        if self.left_hand['gesture'] != Gesture.NONE:
            cv2.putText(img, f'Left Hand: {self.left_hand["gesture"].value}', 
                       (CAMERA_WIDTH - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, YELLOW, 2)
        
        if self.right_hand['gesture'] != Gesture.NONE:
            cv2.putText(img, f'Right Hand: {self.right_hand["gesture"].value}', 
                       (CAMERA_WIDTH - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, YELLOW, 2)
        
        if self.media_action != MediaAction.NONE:
            cv2.putText(img, f'Action: {self.media_action.value}', 
                       (CAMERA_WIDTH - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, GREEN, 2)
            
            # Clear the media action after a short delay
            if time.time() - self.media_action_time > 1.0:
                self.media_action = MediaAction.NONE


def main():
    """Main function to run the hand mouse controller."""
    try:
        # Create and run the controller
        controller = HandMouseController()
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

class HandMouseController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)
        self.frameR = 100  # Frame reduction for better hand detection
        self.smoothening = SMOOTHING
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.last_click_time = 0
        self.last_volume_change = time.time()
        self.media_action = MediaAction.NONE
        self.media_action_time = 0
        self.cursor_radius = 15
        self.cursor_growing = True
        
        # Store hand states
        self.left_hand = {
            'gesture': Gesture.NONE,
            'position': (0, 0),
            'last_gesture': Gesture.NONE,
            'gesture_start_time': 0
        }
        self.right_hand = {
            'position': (0, 0),
        }


        h, w, _ = img_shape
        
        # Invert x-axis and map to screen
        screen_x = np.interp(x, (self.frameR, w - self.frameR), (0, SCREEN_WIDTH))
        screen_y = np.interp(y, (self.frameR, h - self.frameR), (0, SCREEN_HEIGHT))
        
        # Apply smoothing
        self.clocX = self.plocX + (screen_x - self.plocX) / self.smoothening
        self.clocY = self.plocY + (screen_y - self.plocY) / self.smoothening
        
        self.plocX, self.plocY = self.clocX, self.clocY
        
        return int(self.clocX), int(self.clocY)

    def run(self):
        """Main loop for hand tracking and mouse control."""
        p_time = 0
        pinching = False
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from camera")
                break
                
            img = cv2.flip(img, 1)  # Mirror the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(img_rgb)
            
            # Draw rectangle for hand detection area
            cv2.rectangle(img, (self.frameR, self.frameR), 
                         (CAMERA_WIDTH - self.frameR, CAMERA_HEIGHT - self.frameR), 
                         (255, 0, 255), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get index finger position
                    x, y = self.get_finger_position(hand_landmarks, img.shape)
                    
                    # Map to screen coordinates and move mouse
                    screen_x, screen_y = self.map_to_screen(x, y, img.shape)
                    pyautogui.moveTo(screen_x, screen_y)
                    
                    # Check for pinch gesture (click)
                    if self.is_pinching(hand_landmarks, img.shape):
                        current_time = time.time()
                        if not pinching and (current_time - self.last_click_time) > self.click_cooldown:
                            pyautogui.click()
                            self.last_click_time = current_time
                            pinching = True
                            cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)  # Visual feedback
                    else:
                        pinching = False
            
            # Calculate and display FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Display instructions
            cv2.putText(img, 'Pinch to click', (20, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow("Hand Mouse Controller", img)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Create and run the controller
    controller = HandMouseController()
    controller.run()

if __name__ == "__main__":
    main()
