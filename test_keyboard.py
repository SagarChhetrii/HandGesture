"""
Comprehensive Test Suite for MediaPipe Hand Gesture Virtual Keyboard
Tests MediaPipe integration, highlighting, and gesture detection
Python 3.13.6 compatible with modern testing features
"""

import tkinter as tk
import time
import sys
import cv2
import numpy as np
from typing import Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass
import unittest
import threading
import queue
from pathlib import Path

from config import *

try:
    from advanced_keyboard import MediaPipeHandTracker
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available - some tests will be skipped")

@dataclass
class TestResult:
    """Enhanced test result data structure"""
    test_name: str
    passed: bool
    message: str
    duration: float = 0.0
    details: str = ""

class MediaPipeKeyboardTester:
    """Comprehensive tester for MediaPipe integration"""
    
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MediaPipe Keyboard Tester v3.0")
        self.root.geometry("1200x800")
        self.root.configure(bg=BACKGROUND_COLOR)
        
        # Test state
        self.key_buttons: Dict[str, tk.Button] = {}
        self.highlighted_key: Optional[tk.Button] = None
        self.cursor_x: int = 600
        self.cursor_y: int = 400
        self.test_results: List[TestResult] = []
        
        # MediaPipe tracker for testing
        self.hand_tracker: Optional[MediaPipeHandTracker] = None
        self.camera_active: bool = False
        self.test_frame_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Animation state
        self.animation_time: float = 0.0
        self.test_keys: List[str] = ['q', 'w', 'e', 'r', 't', 'SPACE', 'BACKSPACE', 'CAPS', 'NUMBERS']
        
        self.setup_test_gui()
        
    def setup_test_gui(self) -> None:
        """Setup comprehensive test GUI"""
        # Main title
        title = tk.Label(
            self.root,
            text=f"🧪 MediaPipe Hand Tracking Keyboard Tester v3.0",
            font=('Arial', 18, 'bold'),
            bg=BACKGROUND_COLOR,
            fg='white'
        )
        title.pack(pady=15)
        
        # System info
        system_info = tk.Label(
            self.root,
            text=f"🐍 Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | "
                 f"📹 MediaPipe {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not Available'}",
            font=('Arial', 11),
            bg=BACKGROUND_COLOR,
            fg='lightblue'
        )
        system_info.pack(pady=5)
        
        # Main container
        main_container = tk.Frame(self.root, bg=BACKGROUND_COLOR)
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left side - Test controls and camera
        left_panel = tk.Frame(main_container, bg=BACKGROUND_COLOR)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Test control buttons
        control_frame = tk.LabelFrame(
            left_panel,
            text="🎮 Test Controls",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 12, 'bold')
        )
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Control buttons grid
        button_frame = tk.Frame(control_frame, bg=BACKGROUND_COLOR)
        button_frame.pack(pady=10)
        
        test_buttons = [
            ("🔧 Basic Tests", self.run_basic_tests, '#4CAF50'),
            ("⌨️ Keyboard Test", self.test_keyboard_layout, '#2196F3'),
            ("🎯 Cursor Test", self.test_cursor_simulation, '#FF9800'),
            ("📹 Camera Test", self.test_camera_integration, '#9C27B0'),
            ("🤲 MediaPipe Test", self.test_mediapipe_integration, '#00BCD4'),
            ("📊 Performance Test", self.test_performance_metrics, '#795548'),
            ("🔄 Reset Tests", self.reset_all_tests, '#F44336'),
            ("🚀 Run All Tests", self.run_comprehensive_tests, '#E91E63')
        ]
        
        for i, (text, command, color) in enumerate(test_buttons):
            row, col = i // 4, i % 4
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                font=('Arial', 9, 'bold'),
                bg=color,
                fg='white',
                width=15,
                height=2,
                relief='raised',
                bd=2
            )
            btn.grid(row=row, column=col, padx=3, pady=3)
        
        # Camera test area
        camera_frame = tk.LabelFrame(
            left_panel,
            text="📹 Camera Test Area",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 12, 'bold')
        )
        camera_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.camera_label = tk.Label(
            camera_frame,
            bg='black',
            text="📷 Camera Test\n\nClick 'Camera Test' to start\ncamera integration testing",
            fg='white',
            font=('Arial', 12),
            justify='center'
        )
        self.camera_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right side - Keyboard and results
        right_panel = tk.Frame(main_container, bg=BACKGROUND_COLOR)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Cursor position display
        cursor_frame = tk.LabelFrame(
            right_panel,
            text="🎯 Cursor Information",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        cursor_frame.pack(fill='x', pady=(0, 10))
        
        self.cursor_label = tk.Label(
            cursor_frame,
            text=f"Cursor Position: ({self.cursor_x}, {self.cursor_y})",
            font=('Arial', 12, 'bold'),
            bg=BACKGROUND_COLOR,
            fg='cyan'
        )
        self.cursor_label.pack(pady=8)
        
        # Test keyboard
        keyboard_frame = tk.LabelFrame(
            right_panel,
            text="⌨️ Test Keyboard",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        keyboard_frame.pack(fill='x', pady=(0, 10))
        
        self.keyboard_container = tk.Frame(keyboard_frame, bg=BACKGROUND_COLOR)
        self.keyboard_container.pack(pady=10)
        
        self.create_test_keyboard()
        
        # Results display
        results_frame = tk.LabelFrame(
            right_panel,
            text="📋 Test Results",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        results_frame.pack(fill='both', expand=True)
        
        # Results text area with scrollbar
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True, pady=8, padx=8)
        
        self.results_text = tk.Text(
            text_frame,
            font=('Consolas', 9),
            bg='#1e1e1e',
            fg='#00ff00',
            wrap='word',
            height=12
        )
        
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initialize with welcome message
        self.log_result("🚀 MediaPipe Keyboard Tester v3.0 initialized")
        self.log_result("📋 Select a test from the control panel to begin")
        
        # Start cursor simulation
        self.start_cursor_simulation()

    def create_test_keyboard(self) -> None:
        """Create test keyboard layout"""
        # Clear existing keyboard
        for widget in self.keyboard_container.winfo_children():
            widget.destroy()
        
        self.key_buttons.clear()
        
        # Create test keyboard rows
        test_layout = [
            ['q', 'w', 'e', 'r', 't'],
            ['a', 's', 'd', 'f', 'g'],
            ['SPACE', 'BACKSPACE', 'CAPS']
        ]
        
        for row_idx, row in enumerate(test_layout):
            row_frame = tk.Frame(self.keyboard_container, bg=BACKGROUND_COLOR)
            row_frame.pack(pady=2)
            
            for key in row:
                if key == 'SPACE':
                    width, text = 15, '⎵ SPACE'
                elif key == 'BACKSPACE':
                    width, text = 10, '⌫ BACK'
                elif key == 'CAPS':
                    width, text = 8, '🔄 CAPS'
                else:
                    width, text = 6, key
                
                btn = tk.Button(
                    row_frame,
                    text=text,
                    width=width,
                    height=2,
                    font=KEY_FONT,
                    bg=KEY_COLOR,
                    fg='white',
                    relief='raised',
                    bd=2,
                    activebackground=KEY_HOVER_COLOR
                )
                btn.pack(side='left', padx=2)
                self.key_buttons[key] = btn

    def start_cursor_simulation(self) -> None:
        """Start cursor movement simulation"""
        self.update_cursor_simulation()

    def update_cursor_simulation(self) -> None:
        """Update cursor position with smooth animation"""
        self.animation_time += 0.03
        
        # Create smooth movement pattern
        base_x = 600
        base_y = 400
        amplitude_x = 200
        amplitude_y = 50
        
        self.cursor_x = int(base_x + amplitude_x * np.sin(self.animation_time))
        self.cursor_y = int(base_y + amplitude_y * np.cos(self.animation_time * 0.7))
        
        # Update cursor display
        self.cursor_label.config(text=f"Cursor Position: ({self.cursor_x}, {self.cursor_y})")
        
        # Update keyboard highlighting
        self.update_keyboard_highlighting()
        
        # Schedule next update
        self.root.after(50, self.update_cursor_simulation)

    def update_keyboard_highlighting(self) -> None:
        """Update keyboard highlighting based on cursor position"""
        # Reset all key colors
        for btn in self.key_buttons.values():
            btn.config(bg=KEY_COLOR)
        
        # Find key under cursor
        self.highlighted_key = None
        for key, btn in self.key_buttons.items():
            if btn.winfo_viewable():
                try:
                    btn_x = btn.winfo_rootx() - self.root.winfo_rootx()
                    btn_y = btn.winfo_rooty() - self.root.winfo_rooty()
                    btn_w = btn.winfo_width()
                    btn_h = btn.winfo_height()
                    
                    if (btn_x <= self.cursor_x <= btn_x + btn_w and 
                        btn_y <= self.cursor_y <= btn_y + btn_h):
                        self.highlighted_key = btn
                        btn.config(bg=KEY_HOVER_COLOR)
                        break
                except tk.TclError:
                    pass

    def run_basic_tests(self) -> None:
        """Run basic functionality tests"""
        self.log_result("🔧 Starting Basic Functionality Tests...")
        
        tests = [
            ("Configuration Loading", self._test_config_loading),
            ("Keyboard Layout Validation", self._test_keyboard_layouts),
            ("Color Configuration", self._test_color_config),
            ("MediaPipe Availability", self._test_mediapipe_availability),
            ("Python Version Check", self._test_python_version),
            ("Camera Access Test", self._test_camera_access)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = time.time() - start_time
                result = TestResult(test_name, True, f"✅ {test_name} passed", duration)
                passed += 1
            except Exception as e:
                duration = time.time() - start_time
                result = TestResult(test_name, False, f"❌ {test_name} failed: {e}", duration, str(e))
            
            self.test_results.append(result)
            self.log_result(result.message)
        
        self.log_result(f"📊 Basic Tests Summary: {passed}/{total} passed ({(passed/total)*100:.1f}%)")

    def test_keyboard_layout(self) -> None:
        """Test keyboard layout and button creation"""
        self.log_result("⌨️ Testing Keyboard Layout...")
        
        try:
            # Test button creation
            expected_buttons = len(self.test_keys)
            actual_buttons = len(self.key_buttons)
            
            if expected_buttons != actual_buttons:
                raise AssertionError(f"Expected {expected_buttons} buttons, got {actual_buttons}")
            
            # Test button properties
            for key, btn in self.key_buttons.items():
                if not btn.winfo_exists():
                    raise AssertionError(f"Button for key '{key}' does not exist")
                
                # Test button can be configured
                original_bg = btn.cget('bg')
                btn.config(bg='red')
                btn.config(bg=original_bg)
            
            result = TestResult("Keyboard Layout", True, "✅ Keyboard layout test passed")
            self.test_results.append(result)
            self.log_result(result.message)
            
        except Exception as e:
            result = TestResult("Keyboard Layout", False, f"❌ Keyboard layout test failed: {e}")
            self.test_results.append(result)
            self.log_result(result.message)

    def test_cursor_simulation(self) -> None:
        """Test cursor simulation and highlighting"""
        self.log_result("🎯 Testing Cursor Simulation...")
        
        test_positions = [
            (500, 350), (600, 350), (700, 350),
            (500, 400), (600, 400), (700, 400)
        ]
        
        successful_highlights = 0
        total_tests = len(test_positions)
        
        for i, (x, y) in enumerate(test_positions):
            self.cursor_x, self.cursor_y = x, y
            self.update_keyboard_highlighting()
            
            if self.highlighted_key is not None:
                successful_highlights += 1
                self.log_result(f"  Position {i+1}: ({x}, {y}) - Key highlighted")
            else:
                self.log_result(f"  Position {i+1}: ({x}, {y}) - No highlight")
            
            time.sleep(0.1)
        
        success_rate = (successful_highlights / total_tests) * 100
        
        if success_rate >= 60:  # Lower threshold for test environment
            result = TestResult("Cursor Simulation", True, 
                              f"✅ Cursor test passed: {successful_highlights}/{total_tests} highlights ({success_rate:.1f}%)")
        else:
            result = TestResult("Cursor Simulation", False,
                              f"❌ Cursor test failed: {successful_highlights}/{total_tests} highlights ({success_rate:.1f}%)")
        
        self.test_results.append(result)
        self.log_result(result.message)

    def test_camera_integration(self) -> None:
        """Test camera integration"""
        self.log_result("📹 Testing Camera Integration...")
        
        try:
            # Test camera access
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
            
            # Test frame reading
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read from camera")
            
            # Test frame properties
            if frame is None or frame.size == 0:
                raise RuntimeError("Invalid frame received")
            
            height, width = frame.shape[:2]
            self.log_result(f"  Camera resolution: {width}x{height}")
            
            # Test multiple frame reads
            frames_read = 0
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_read += 1
                time.sleep(0.1)
            
            cap.release()
            
            if frames_read < 3:
                raise RuntimeError(f"Only {frames_read}/5 frames read successfully")
            
            result = TestResult("Camera Integration", True, 
                              f"✅ Camera test passed: {frames_read}/5 frames read")
            self.test_results.append(result)
            self.log_result(result.message)
            
        except Exception as e:
            result = TestResult("Camera Integration", False, f"❌ Camera test failed: {e}")
            self.test_results.append(result)
            self.log_result(result.message)

    def test_mediapipe_integration(self) -> None:
        """Test MediaPipe integration"""
        self.log_result("🤲 Testing MediaPipe Integration...")
        
        if not MEDIAPIPE_AVAILABLE:
            result = TestResult("MediaPipe Integration", False, "❌ MediaPipe not available")
            self.test_results.append(result)
            self.log_result(result.message)
            return
        
        try:
            # Initialize MediaPipe tracker
            tracker = MediaPipeHandTracker()
            
            # Create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame.fill(128)  # Gray background
            
            # Process test frame
            processed_frame, pinch_detected = tracker.process_frame(test_frame)
            
            # Verify processed frame
            if processed_frame is None or processed_frame.size == 0:
                raise RuntimeError("MediaPipe returned invalid frame")
            
            # Test MediaPipe hands initialization
            if not hasattr(tracker, 'hands') or tracker.hands is None:
                raise RuntimeError("MediaPipe hands not properly initialized")
            
            # Test cursor position
            cursor_pos = tracker.get_cursor_position()
            if not isinstance(cursor_pos, tuple) or len(cursor_pos) != 2:
                raise RuntimeError("Invalid cursor position format")
            
            # Test detection status
            status = tracker.get_detection_status()
            if not isinstance(status, str):
                raise RuntimeError("Invalid detection status format")
            
            # Cleanup
            tracker.cleanup()
            
            result = TestResult("MediaPipe Integration", True, "✅ MediaPipe integration test passed")
            self.test_results.append(result)
            self.log_result(result.message)
            
        except Exception as e:
            result = TestResult("MediaPipe Integration", False, f"❌ MediaPipe test failed: {e}")
            self.test_results.append(result)
            self.log_result(result.message)

    def test_performance_metrics(self) -> None:
        """Test performance metrics"""
        self.log_result("📊 Testing Performance Metrics...")
        
        if not MEDIAPIPE_AVAILABLE:
            self.log_result("❌ MediaPipe not available - skipping performance test")
            return
        
        try:
            tracker = MediaPipeHandTracker()
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Performance test
            start_time = time.time()
            iterations = 50
            
            for i in range(iterations):
                processed_frame, pinch_detected = tracker.process_frame(test_frame)
                if i % 10 == 0:
                    self.log_result(f"  Processing iteration {i+1}/{iterations}")
            
            duration = time.time() - start_time
            fps = iterations / duration
            
            tracker.cleanup()
            
            # Performance criteria
            if fps >= 15:
                result = TestResult("Performance", True, 
                                  f"✅ Performance test passed: {fps:.1f} FPS ({duration:.2f}s for {iterations} frames)")
            else:
                result = TestResult("Performance", False,
                                  f"❌ Performance test failed: {fps:.1f} FPS (minimum 15 FPS required)")
            
            self.test_results.append(result)
            self.log_result(result.message)
            
        except Exception as e:
            result = TestResult("Performance", False, f"❌ Performance test failed: {e}")
            self.test_results.append(result)
            self.log_result(result.message)

    def run_comprehensive_tests(self) -> None:
        """Run all tests in sequence"""
        self.log_result("🚀 Starting Comprehensive Test Suite...")
        self.log_result("=" * 50)
        
        # Clear previous results
        self.test_results.clear()
        
        # Run all test categories
        test_categories = [
            self.run_basic_tests,
            self.test_keyboard_layout,
            self.test_cursor_simulation,
            self.test_camera_integration,
            self.test_mediapipe_integration,
            self.test_performance_metrics
        ]
        
        for test_func in test_categories:
            try:
                test_func()
                self.log_result("-" * 30)
            except Exception as e:
                self.log_result(f"❌ Test category failed: {e}")
        
        # Generate comprehensive summary
        self._generate_test_summary()

    def _generate_test_summary(self) -> None:
        """Generate comprehensive test summary"""
        if not self.test_results:
            self.log_result("❌ No test results available")
            return
        
        passed_tests = sum(1 for result in self.test_results if result.passed)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        total_duration = sum(result.duration for result in self.test_results)
        
        summary = f"""
📊 COMPREHENSIVE TEST SUMMARY
{'='*50}
✅ Passed: {passed_tests}
❌ Failed: {total_tests - passed_tests}
📈 Success Rate: {success_rate:.1f}%
⏱️ Total Duration: {total_duration:.2f}s
🧪 Total Tests: {total_tests}
🐍 Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
📹 MediaPipe: {'Available' if MEDIAPIPE_AVAILABLE else 'Not Available'}
{'='*50}

FAILED TESTS:
"""
        
        failed_tests = [r for r in self.test_results if not r.passed]
        if failed_tests:
            for result in failed_tests:
                summary += f"❌ {result.test_name}: {result.details or 'No details'}\n"
        else:
            summary += "🎉 No failed tests!\n"
        
        summary += "=" * 50
        
        self.log_result(summary)

    def reset_all_tests(self) -> None:
        """Reset all test results and state"""
        self.test_results.clear()
        self.results_text.delete(1.0, tk.END)
        
        # Reset keyboard colors
        for btn in self.key_buttons.values():
            btn.config(bg=KEY_COLOR)
        
        self.highlighted_key = None
        
        self.log_result("🔄 All tests reset")
        self.log_result("📋 Ready for new test execution")

    def log_result(self, message: str) -> None:
        """Log result to display with timestamp"""
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.results_text.insert(tk.END, formatted_message)
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    # Helper test methods
    def _test_config_loading(self) -> None:
        """Test configuration loading"""
        assert WINDOW_WIDTH > 0 and WINDOW_HEIGHT > 0
        assert len(KEYBOARD_LAYOUTS) > 0
        assert PINCH_THRESHOLD > 0

    def _test_keyboard_layouts(self) -> None:
        """Test keyboard layout validation"""
        for mode, layout in KEYBOARD_LAYOUTS.items():
            assert len(layout) > 0
            for row in layout:
                assert len(row) > 0

    def _test_color_config(self) -> None:
        """Test color configuration"""
        colors = [BACKGROUND_COLOR, KEY_COLOR, KEY_HOVER_COLOR, KEY_PRESSED_COLOR]
        for color in colors:
            assert color.startswith('#') and len(color) == 7

    def _test_mediapipe_availability(self) -> None:
        """Test MediaPipe availability"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not available")
        
        import mediapipe as mp
        assert hasattr(mp.solutions, 'hands')

    def _test_python_version(self) -> None:
        """Test Python version compatibility"""
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"

    def _test_camera_access(self) -> None:
        """Test basic camera access"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Try alternate camera indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
            else:
                raise RuntimeError("No camera available")
        cap.release()

    def run(self) -> None:
        """Run the test application"""
        print("🧪 Starting MediaPipe Keyboard Tester v3.0...")
        print("📋 This comprehensive tester validates all system components")
        print("🎯 Use the control panel to run specific tests or full suite")
        print(f"🐍 Running on Python {sys.version}")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Test interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")

def run_command_line_tests() -> None:
    """Run basic tests from command line without GUI"""
    print("=" * 60)
    print("🔧 COMMAND LINE BASIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", lambda: _validate_config()),
        ("MediaPipe Availability", lambda: _check_mediapipe()),
        ("Camera Access", lambda: _check_camera()),
        ("Python Version", lambda: _check_python_version())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            test_func()
            print(f"   ✅ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print(f"🐍 Python Version: {sys.version}")
    
    if passed == total:
        print("🎉 All basic tests passed! Ready to run full application.")
    else:
        print("⚠️ Some tests failed. Check dependencies and system setup.")
    
    print("🚀 Run GUI test for interactive testing...")
    print("="*60)

def _validate_config() -> None:
    """Validate configuration"""
    assert WINDOW_WIDTH > 0
    assert len(KEYBOARD_LAYOUTS) > 0

def _check_mediapipe() -> None:
    """Check MediaPipe availability"""
    import mediapipe as mp
    assert hasattr(mp.solutions, 'hands')

def _check_camera() -> None:
    """Check camera access"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")
    cap.release()

def _check_python_version() -> None:
    """Check Python version"""
    assert sys.version_info >= (3, 8)

if __name__ == "__main__":
    print("🧪 MediaPipe Keyboard Tester v3.0 - Choose test mode:")
    print("1. 🔧 Command line basic tests")
    print("2. 🖥️ Interactive GUI test suite")
    print("3. 🚀 Full comprehensive testing")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            run_command_line_tests()
        elif choice == "2":
            tester = MediaPipeKeyboardTester()
            tester.run()
        elif choice == "3":
            run_command_line_tests()
            input("\nPress Enter to continue to GUI testing...")
            tester = MediaPipeKeyboardTester()
            tester.run()
        else:
            print("Invalid choice. Running basic tests...")
            run_command_line_tests()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        print("Make sure you have: pip install mediapipe opencv-python pillow")