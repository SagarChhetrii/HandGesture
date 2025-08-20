✋ Hand Gesture Virtual Keyboard v2.1

A Python-based virtual keyboard that uses two-hand gesture control for typing without a physical keyboard. Built for Python 3.13.6 with OpenCV-based hand tracking.

🚀 Features

Two-Hand Control:

👉 Right Hand (Pointer) → Moves the on-screen cursor with the index finger tip

👌 Left Hand (Pinch) → Types the currently highlighted key

Pinch-to-Type Gesture:

Single pinch = type selected key

Works with letters, numbers, space, backspace, and special keys

Visual Hand Tracking:

Red points on all detected landmarks

Green connecting lines between fingers (like real hand skeleton overlay)

Smooth, real-time webcam feed

Multiple Keyboard Layouts:

Lowercase

Uppercase

Numbers/Symbols

Special keys: SPACE, BACKSPACE, CAPS, NUMBERS, CLEAR

Debounce Protection: Prevents multiple key presses from one pinch (0.3s delay)

Lightweight & Modern GUI: Built with Tkinter

🛠️ Requirements

Python: 3.13.6 (required)

OS: Windows 10/11, macOS, or Linux

Webcam: For hand tracking

Libraries: See requirements.txt

📦 Installation
1. Clone Project
git clone <repository-url>
cd py

2. Create Virtual Environment
python -m venv venv

3. Activate Virtual Environment

Windows:

.\venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

🚀 Usage

Start the app:

python main.py


Show both hands in front of webcam (1–2 feet away)

Move Cursor: Right-hand index finger

Type Key: Left-hand pinch gesture

🎯 Example Controls

👉 Right Index Finger → Cursor moves on virtual keyboard

👌 Left Hand Pinch → Types selected key

🔡 Switch Layouts: CAPS, NUMBERS buttons available on-screen

⌫ Backspace: Highlight key → pinch

🧪 Testing

Run test suite:

python test_keyboard.py


Tests:

✅ Webcam feed detection

✅ OpenCV hand tracking

✅ Tkinter GUI responsiveness

✅ Key highlight & pinch typing

⚙️ Configuration

Edit config.py for:

CAMERA_WIDTH (default: 640)

CAMERA_HEIGHT (default: 480)

CAMERA_FPS (default: 30)

PINCH_THRESHOLD (adjust pinch sensitivity)

🔧 Troubleshooting

Webcam not detected

Check camera permissions

Use correct index (0, 1, 2)

Hand not detected properly

Improve lighting

Keep background simple

Adjust sensitivity in config

Multiple presses per pinch

Increase debounce delay in config.py

📁 Project Structure
py/
├── main.py                # Main app entry
├── advanced_keyboard.py   # Virtual keyboard + layout logic
├── config.py              # Config and constants
├── test_keyboard.py       # Tests
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

🔄 Updates

✅ Replaced MediaPipe with pure OpenCV (Python 3.13.6 compatible)

✅ Removed extra gestures (fist, palm, peace sign)

✅ Improved two-hand logic (Pointer + Typing hand)

✅ Better debounce handling

📄 License

Open source under the MIT License.

✨ Now you can type with just your hands – no keyboard required!