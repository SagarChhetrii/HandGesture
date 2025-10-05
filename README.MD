✋ Hand Gesture Virtual Keyboard v3.0

A Python-based virtual keyboard that uses two-hand gesture control for typing without a physical keyboard. Supports:

- Python 3.12: Full MediaPipe hand tracking
- Python 3.13: OpenCV-only fallback (no hand tracking)

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

Python: 3.12.7 recommended (pinned via `.python-version`)

OS: Windows 10/11, macOS, or Linux

Webcam: For hand tracking

Libraries: See requirements.txt

📦 Quick Start (macOS)

Option A — Full tracking (recommended):
```bash
cd /Users/sagarchhetri/Desktop/HandGesture
source .venv312/bin/activate
python -m pip install -r requirements.txt  # if needed
python main.py
```

Option B — Python 3.13 fallback (no tracking):
```bash
cd /Users/sagarchhetri/Desktop/HandGesture
source .venv/bin/activate
export TK_SILENCE_DEPRECATION=1
export PKG_CONFIG_PATH=/opt/homebrew/opt/tcl-tk/lib/pkgconfig:${PKG_CONFIG_PATH:-}
python main.py
```

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
- System Settings → Privacy & Security → Camera → enable for Terminal and Python
- Ensure no app is using the camera
- Try indices 0..3 in `config.py`

Hand not detected properly
- Improve lighting and reduce background clutter
- Keep both hands 1–2 feet from camera
- Adjust `PINCH_THRESHOLD` in `config.py`

Multiple presses per pinch
- Increase `PINCH_DEBOUNCE_TIME` in `config.py`

📁 Project Structure
py/
├── main.py                # Main app entry
├── advanced_keyboard.py   # Virtual keyboard + layout logic
├── config.py              # Config and constants
├── test_keyboard.py       # Tests
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

🔄 Updates

✅ MediaPipe on Python 3.12 (full tracking)
✅ OpenCV-only fallback on Python 3.13

✅ Removed extra gestures (fist, palm, peace sign)

✅ Improved two-hand logic (Pointer + Typing hand)

✅ Better debounce handling

📄 License

Open source under the MIT License.

✨ Now you can type with just your hands – no keyboard required!