# ZeroTouch

**ZeroTouch** is a futuristic, AI-powered gesture control application that turns your hands into a mouse! 👋🖱️

📘 **[Read the User Guide](USER_GUIDE.md)** for detailed instructions.

## Features
- **Touchless Control**: navigate your computer without touching a mouse.
- **Dual Hand System**: Separte hands for moving and clicking for better precision.
- Real-time hand tracking using MediaPipe
- Smooth mouse pointer movement
- Pinch gesture for left-click
- Adjustable sensitivity and smoothing
- Works with any standard webcam
- Cross-platform support (Windows, macOS, Linux)

## Requirements

- Python 3.7+
- Webcam
- Windows 10/11, macOS, or Linux

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your webcam is connected and working
2. Run the application:
   ```
   python hand_mouse.py
   ```
3. Position your hand in front of the camera
4. Move your index finger to control the mouse pointer
5. Pinch your thumb and index finger together to click
6. Press 'q' to quit the application

## Controls

- **Move Hand**: Move your index finger to control the mouse pointer
- **Left Click**: Pinch your thumb and index finger together
- **Quit**: Press 'q' in the camera window

## Customization

You can adjust the following parameters in the code:

- `SENSITIVITY`: Adjust mouse movement sensitivity (default: 1.5)
- `SMOOTHING`: Adjust smoothing of mouse movement (default: 5)
- `PINCH_THRESHOLD`: Adjust pinch detection sensitivity (default: 40)
- `click_cooldown`: Adjust time between clicks (default: 0.3s)

## Troubleshooting

- If the mouse movement is too fast/slow, adjust the `SENSITIVITY` value
- If the mouse pointer is jittery, increase the `SMOOTHING` value
- If clicks aren't registering, decrease the `PINCH_THRESHOLD` value
- Make sure your hand is well-lit and clearly visible to the webcam

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control
