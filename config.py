"""
Configuration constants for Hand Gesture Virtual Keyboard
Python 3.13.6 compatible with MediaPipe integration
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

# Camera settings
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 30

# MediaPipe settings
MEDIAPIPE_CONFIDENCE: float = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.5
MAX_NUM_HANDS: int = 2

# Hand tracking settings
PINCH_THRESHOLD: float = 30.0  # Distance threshold for pinch detection (pixels)
PINCH_DEBOUNCE_TIME: float = 0.3  # Seconds between pinch detections
SMOOTHING_ALPHA: float = 0.7  # Cursor smoothing factor (0.0 = no smoothing, 1.0 = instant)

# Landmark indices (MediaPipe standard)
THUMB_TIP: int = 4
INDEX_TIP: int = 8
MIDDLE_TIP: int = 12
RING_TIP: int = 16
PINKY_TIP: int = 20
WRIST: int = 0

# Hand roles
RIGHT_HAND_LABEL: str = "Right"  # Pointer hand (cursor control)
LEFT_HAND_LABEL: str = "Left"    # Typing hand (pinch to click)

# Window dimensions
WINDOW_WIDTH: int = 1400
WINDOW_HEIGHT: int = 900

# Colors (RGB tuples for OpenCV, hex strings for Tkinter)
LANDMARK_COLOR: Tuple[int, int, int] = (0, 0, 255)      # Red dots (BGR format)
CONNECTION_COLOR: Tuple[int, int, int] = (0, 255, 0)    # Green lines (BGR format)
CURSOR_COLOR: Tuple[int, int, int] = (255, 0, 255)      # Magenta cursor (BGR format)
PINCH_COLOR: Tuple[int, int, int] = (0, 255, 255)       # Yellow pinch line (BGR format)

# Tkinter colors
BACKGROUND_COLOR: str = '#2c2c2c'      # Dark background
TEXT_AREA_COLOR: str = '#ffffff'       # Pure white text area
KEY_COLOR: str = '#4a4a4a'            # Default key color
KEY_HOVER_COLOR: str = '#6a9bd1'      # Blue hover color for cursor
KEY_PRESSED_COLOR: str = '#ff6b6b'    # Red pressed color
STATUS_COLOR: str = '#44ff44'         # Green status
CURSOR_DISPLAY_COLOR: str = '#ff4444' # Red cursor display

# Fonts
TEXT_FONT: Tuple[str, int] = ('Consolas', 14)
STATUS_FONT: Tuple[str, int] = ('Arial', 11)
KEY_FONT: Tuple[str, int, str] = ('Arial', 10, 'bold')

# Keyboard layout settings
KEY_WIDTH: int = 60
KEY_HEIGHT: int = 50
KEY_SPACING: int = 5
CURSOR_BOUNDARY_MARGIN: int = 20

# Keyboard layouts
KEYBOARD_LAYOUTS: Dict[str, List[List[str]]] = {
    'lowercase': [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
        ['SPACE', 'BACKSPACE', 'CAPS', 'NUMBERS']
    ],
    'uppercase': [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
        ['SPACE', 'BACKSPACE', 'CAPS', 'NUMBERS']
    ],
    'numbers': [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['+', '-', '*', '/', '=', '(', ')', '.', ',', '?'],
        ['!', '@', '#', '$', '%', '^', '&', ':', ';', '"'],
        ['SPACE', 'BACKSPACE', 'LETTERS', 'CLEAR']
    ]
}

@dataclass
class CursorPosition:
    """Dataclass for cursor position with bounds checking"""
    x: int
    y: int
    
    def __post_init__(self) -> None:
        """Ensure cursor stays within bounds"""
        self.x = max(CURSOR_BOUNDARY_MARGIN, min(WINDOW_WIDTH - CURSOR_BOUNDARY_MARGIN, self.x))
        self.y = max(CURSOR_BOUNDARY_MARGIN, min(WINDOW_HEIGHT - CURSOR_BOUNDARY_MARGIN, self.y))

@dataclass
class HandLandmark:
    """MediaPipe hand landmark point"""
    x: int
    y: int
    z: float = 0.0
    visibility: float = 1.0

@dataclass
class DetectedHand:
    """Detected hand information"""
    landmarks: List[HandLandmark]
    handedness: str  # "Left" or "Right"
    confidence: float

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        assert WINDOW_WIDTH > 0 and WINDOW_HEIGHT > 0, "Window dimensions must be positive"
        assert PINCH_THRESHOLD > 0, "Pinch threshold must be positive"
        assert 0.0 < SMOOTHING_ALPHA <= 1.0, "Smoothing alpha must be between 0 and 1"
        assert len(KEYBOARD_LAYOUTS) > 0, "At least one keyboard layout required"
        assert MAX_NUM_HANDS >= 2, "Must support at least 2 hands"
        
        # Validate colors are hex format for Tkinter
        colors = [BACKGROUND_COLOR, TEXT_AREA_COLOR, KEY_COLOR, KEY_HOVER_COLOR, KEY_PRESSED_COLOR]
        for color in colors:
            assert color.startswith('#') and len(color) == 7, f"Invalid color format: {color}"
        
        return True
    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False

# Validate configuration on import
if not validate_config():
    raise RuntimeError("Invalid configuration detected")