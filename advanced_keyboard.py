"""
Advanced Hand Gesture Detection Module
MediaPipe-based dual-hand tracking for Python 3.13.6 compatibility
Right hand for cursor control, Left hand for pinch clicking
"""

import cv2
import numpy as np
try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    _HAS_MEDIAPIPE = False
import time
from typing import Optional, Tuple, List, Dict
import math

from config import *

if _HAS_MEDIAPIPE:

    class MediaPipeHandTracker:
        """MediaPipe-based dual-hand tracker for Python 3.13.6"""
        
        def __init__(self) -> None:
            """Initialize MediaPipe hands solution"""
            print("[HandTracker] Initializing MediaPipe hand tracking...")
            
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Configure hands detection
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=MEDIAPIPE_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
                model_complexity=1  # Higher accuracy model
            )
            
            # Hand tracking state
            self.right_hand: Optional[DetectedHand] = None
            self.left_hand: Optional[DetectedHand] = None
            
            # Cursor state
            self.cursor_position = CursorPosition(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
            self.last_cursor_x: Optional[int] = None
            self.last_cursor_y: Optional[int] = None
            
            # Pinch detection state
            self.is_pinching: bool = False
            self.last_pinch_time: float = 0.0
            self.pinch_debounce: float = PINCH_DEBOUNCE_TIME
            
            # Performance tracking
            self.frame_count: int = 0
            self.last_fps_time: float = time.time()
            self.current_fps: float = 0.0
            
            print("[HandTracker] MediaPipe initialization complete")
        
        def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
            """
            Process frame with MediaPipe hand detection
            
            Args:
                frame: Input camera frame
                
            Returns:
                Tuple of (processed_frame, pinch_detected)
            """
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            # Reset hand states
            self.right_hand = None
            self.left_hand = None
            
            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                self._process_detected_hands(frame, results)
            
            # Update cursor position from right hand
            self._update_cursor_from_right_hand(frame)
            
            # Check for pinch gesture from left hand
            pinch_detected = self._check_left_hand_pinch(frame)
            
            # Draw status information
            self._draw_status_info(frame)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.frame_count += 1
            return frame, pinch_detected

        def _process_detected_hands(self, frame: np.ndarray, results) -> None:
            """Process MediaPipe detection results and extract hand information"""
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (MediaPipe returns "Left"/"Right" from camera perspective)
                hand_label = handedness.classification[0].label
                # Mirror correction depending on setting
                if AUTO_MIRROR_SWAP:
                    effective_label = "Right" if hand_label == "Left" else "Left"
                else:
                    effective_label = hand_label
                confidence = handedness.classification[0].score

                # Convert landmarks to our format
                landmarks: List[HandLandmark] = []
                h, w = frame.shape[:2]

                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z if hasattr(landmark, 'z') else 0.0
                    visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0

                    landmarks.append(HandLandmark(x, y, z, visibility))

                # Create detected hand object with effective handedness
                detected_hand = DetectedHand(landmarks, effective_label, confidence)

                # Apply optional manual swap
                final_label = effective_label
                if SWAP_HANDS:
                    final_label = "Left" if effective_label == "Right" else "Right"

                # Assign to appropriate hand after corrections
                if final_label == "Right":
                    self.right_hand = detected_hand
                    self._draw_hand_skeleton(frame, hand_landmarks, "RIGHT (Pointer)", (0, 255, 255))
                else:  # "Left"
                    self.left_hand = detected_hand
                    self._draw_hand_skeleton(frame, hand_landmarks, "LEFT (Clicker)", (255, 255, 0))

        def _draw_hand_skeleton(self, frame: np.ndarray, hand_landmarks, label: str, color: Tuple[int, int, int]) -> None:
            """Draw MediaPipe hand skeleton with red dots and green lines"""
            h, w = frame.shape[:2]
            
            # Draw connections (green lines)
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=CONNECTION_COLOR, thickness=2, circle_radius=1),
                self.mp_draw.DrawingSpec(color=CONNECTION_COLOR, thickness=2)
            )
            
            # Draw landmarks (red dots)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw red circles for all landmarks
                cv2.circle(frame, (x, y), 4, LANDMARK_COLOR, -1)
                
                # Highlight fingertips with larger circles
                if idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
                    cv2.circle(frame, (x, y), 6, LANDMARK_COLOR, 2)
            
            # Draw hand label
            if hand_landmarks.landmark:
                wrist = hand_landmarks.landmark[WRIST]
                label_x = int(wrist.x * w)
                label_y = int(wrist.y * h) - 30
                
                cv2.putText(frame, label, (label_x - 60, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        def _update_cursor_from_right_hand(self, frame: np.ndarray) -> None:
            """Update cursor position based on right hand index finger tip"""
            if not self.right_hand or len(self.right_hand.landmarks) <= INDEX_TIP:
                return
            
            # Get index finger tip position
            index_tip = self.right_hand.landmarks[INDEX_TIP]
            finger_x, finger_y = index_tip.x, index_tip.y
            
            # Map camera coordinates to window coordinates
            frame_h, frame_w = frame.shape[:2]
            
            # Map camera coords to full application window coordinates
            # Tk places camera on left panel and keyboard on right; use full window for hit-testing,
            # since GUI code computes absolute positions relative to root window.
            keyboard_x = int((finger_x / frame_w) * WINDOW_WIDTH)
            keyboard_y = int((finger_y / frame_h) * WINDOW_HEIGHT)
            
            # Apply smoothing to reduce jitter
            if self.last_cursor_x is not None and self.last_cursor_y is not None:
                keyboard_x = int(SMOOTHING_ALPHA * keyboard_x + (1 - SMOOTHING_ALPHA) * self.last_cursor_x)
                keyboard_y = int(SMOOTHING_ALPHA * keyboard_y + (1 - SMOOTHING_ALPHA) * self.last_cursor_y)
            
            # Update cursor position with bounds checking
            self.cursor_position = CursorPosition(keyboard_x, keyboard_y)
            self.last_cursor_x = self.cursor_position.x
            self.last_cursor_y = self.cursor_position.y
            
            # Draw cursor indicator on camera frame
            cv2.circle(frame, (finger_x, finger_y), 8, CURSOR_COLOR, -1)
            cv2.putText(frame, f"CURSOR", (finger_x - 30, finger_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CURSOR_COLOR, 2)

        def _check_left_hand_pinch(self, frame: np.ndarray) -> bool:
            """Check for pinch gesture using left hand thumb and index finger"""
            if not self.left_hand or len(self.left_hand.landmarks) <= INDEX_TIP:
                return False
            
            # Get thumb tip and index finger tip
            thumb_tip = self.left_hand.landmarks[THUMB_TIP]
            index_tip = self.left_hand.landmarks[INDEX_TIP]
            
            # Calculate Euclidean distance between thumb and index finger tips
            distance = math.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 + 
                (thumb_tip.y - index_tip.y) ** 2
            )
            
            # Draw pinch visualization line
            cv2.line(frame, (thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y), 
                    PINCH_COLOR, 3)
            
            # Display distance
            mid_x = (thumb_tip.x + index_tip.x) // 2
            mid_y = (thumb_tip.y + index_tip.y) // 2
            cv2.putText(frame, f"Pinch: {distance:.0f}px", (mid_x - 50, mid_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINCH_COLOR, 2)
            
            # Check for pinch with debounce
            current_time = time.time()
            
            if (distance < PINCH_THRESHOLD and 
                not self.is_pinching and 
                (current_time - self.last_pinch_time) > self.pinch_debounce):
                
                self.is_pinching = True
                self.last_pinch_time = current_time
                
                # Visual feedback for pinch detection
                cv2.putText(frame, "PINCH DETECTED!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw highlight box around detected pinch
                cv2.rectangle(frame, (thumb_tip.x - 20, thumb_tip.y - 20), 
                             (thumb_tip.x + 20, thumb_tip.y + 20), (0, 0, 255), 3)
                
                print(f"[HandTracker] Pinch detected! Distance: {distance:.1f}px")
                return True
                
            elif distance >= PINCH_THRESHOLD + 10:  # Add hysteresis to prevent flickering
                self.is_pinching = False
            
            return False

        def _draw_status_info(self, frame: np.ndarray) -> None:
            """Draw comprehensive status information on camera frame"""
            # Status background
            cv2.rectangle(frame, (10, 10), (450, 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (450, 140), (0, 255, 0), 2)
            
            y_offset = 35
            
            # Hand detection status
            right_status = "✅ DETECTED" if self.right_hand else "❌ NOT FOUND"
            left_status = "✅ DETECTED" if self.left_hand else "❌ NOT FOUND"
            
            cv2.putText(frame, f"RIGHT HAND (Pointer): {right_status}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            cv2.putText(frame, f"LEFT HAND (Clicker): {left_status}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            # Cursor position
            if self.right_hand:
                cv2.putText(frame, f"CURSOR: ({self.cursor_position.x}, {self.cursor_position.y})", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Performance info
            cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Frame: {self.frame_count}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        def _update_performance_metrics(self) -> None:
            """Update FPS and performance metrics"""
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
                self.current_fps = self.frame_count / (current_time - self.last_fps_time + 0.001)
                self.last_fps_time = current_time
                self.frame_count = 0

        def get_cursor_position(self) -> Tuple[int, int]:
            """Get current cursor position as tuple"""
            return self.cursor_position.x, self.cursor_position.y

        def is_right_hand_detected(self) -> bool:
            """Check if right hand (pointer) is detected"""
            return self.right_hand is not None

        def is_left_hand_detected(self) -> bool:
            """Check if left hand (clicker) is detected"""
            return self.left_hand is not None

        def get_detection_status(self) -> str:
            """Get formatted detection status string"""
            right = "✅" if self.is_right_hand_detected() else "❌"
            left = "✅" if self.is_left_hand_detected() else "❌"
            return f"Pointer: {right} | Clicker: {left}"

        def get_hand_info(self) -> Dict[str, any]:
            """Get detailed hand information for debugging"""
            info = {
                'right_hand_detected': self.is_right_hand_detected(),
                'left_hand_detected': self.is_left_hand_detected(),
                'cursor_position': self.get_cursor_position(),
                'is_pinching': self.is_pinching,
                'fps': self.current_fps,
                'frame_count': self.frame_count
            }
            
            if self.right_hand:
                info['right_hand_confidence'] = self.right_hand.confidence
                info['right_hand_landmarks'] = len(self.right_hand.landmarks)
            
            if self.left_hand:
                info['left_hand_confidence'] = self.left_hand.confidence
                info['left_hand_landmarks'] = len(self.left_hand.landmarks)
            
            return info

        def cleanup(self) -> None:
            """Cleanup MediaPipe resources"""
            print("[HandTracker] Cleaning up MediaPipe resources...")
            if hasattr(self, 'hands'):
                self.hands.close()
            print("[HandTracker] Cleanup complete")
else:

    class MediaPipeHandTracker:
        """OpenCV-only fallback when MediaPipe isn't available.

        This keeps the same public API so the GUI can run on Python 3.13.
        The fallback does NOT perform real hand tracking; it draws guidance
        and keeps cursor in the last position.
        """
        
        def __init__(self) -> None:
            print("[HandTracker] MediaPipe not available; using OpenCV fallback (no tracking)")
            
            # State placeholders to match the expected interface
            self.right_hand: Optional[DetectedHand] = None
            self.left_hand: Optional[DetectedHand] = None
            self.cursor_position = CursorPosition(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
            self.last_cursor_x: Optional[int] = None
            self.last_cursor_y: Optional[int] = None
            self.is_pinching: bool = False
            self.last_pinch_time: float = 0.0
            self.pinch_debounce: float = PINCH_DEBOUNCE_TIME
            self.frame_count: int = 0
            self.last_fps_time: float = time.time()
            self.current_fps: float = 0.0

        def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
            """Pass-through frame with helper overlays; no hand tracking."""
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            # Draw helper panel
            cv2.rectangle(frame, (10, 10), (max(10, min(500, w - 10)), 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (max(10, min(500, w - 10)), 130), (0, 255, 255), 2)
            y = 35
            cv2.putText(frame, "MediaPipe not available", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); y += 25
            cv2.putText(frame, "Install mediapipe (when Py3.13 wheels ship)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2); y += 22
            cv2.putText(frame, "Fallback active: no hand tracking", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2); y += 22

            # Show current (static) cursor position for GUI alignment
            cv2.circle(frame, (self.cursor_position.x, self.cursor_position.y), 6, CURSOR_COLOR, 2)
            cv2.putText(frame, f"CURSOR ({self.cursor_position.x}, {self.cursor_position.y})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Update FPS estimate
            self._update_performance_metrics()
            self.frame_count += 1
            return frame, False

        # The following methods are no-ops to preserve the public API
        def _update_performance_metrics(self) -> None:
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time + 0.001)
                self.last_fps_time = current_time
                self.frame_count = 0

        def get_cursor_position(self) -> Tuple[int, int]:
            return self.cursor_position.x, self.cursor_position.y

        def is_right_hand_detected(self) -> bool:
            return False

        def is_left_hand_detected(self) -> bool:
            return False

        def get_detection_status(self) -> str:
            return "Pointer: ❌ | Clicker: ❌"

        def cleanup(self) -> None:
            print("[HandTracker] Fallback cleanup complete")
    

# Alias for backward compatibility
DualHandTracker = MediaPipeHandTracker