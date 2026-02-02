import cv2
import pyautogui
import numpy as np
import time

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
        self.cursor_radius = 15
        self.cursor_color = GREEN
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.is_clicking = False
        self.click_start_time = 0

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
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Let the background subtractor learn the background
        print("Learning background... Please wait...")
        for _ in range(60):
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture background")
                return
            cv2.waitKey(30)
        print("Background learning complete!")
        
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Error: Failed to capture image from camera")
                    break
                    
                # Make sure we have a valid image
                if img is None:
                    print("Error: Received empty frame from camera")
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
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Clean up
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("Camera released. Goodbye!")

def main():
    # Create and run the controller
    controller = HandMouseController()
    controller.run()

if __name__ == "__main__":
    main()
