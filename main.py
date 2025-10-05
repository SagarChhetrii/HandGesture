"""
Main Hand Gesture Virtual Keyboard Application
MediaPipe + OpenCV + Tkinter integration for Python 3.13.6
Dual-hand tracking: Right hand for cursor, Left hand for clicking
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import numpy as np
from PIL import Image, ImageTk
import time
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import sys
from pathlib import Path

from config import *
from advanced_keyboard import MediaPipeHandTracker

@dataclass
class AppState:
    """Application state management"""
    current_mode: str = 'lowercase'
    typed_text: str = ""
    key_count: int = 0
    start_time: float = field(default_factory=time.time)
    running: bool = True
    pinch_count: int = 0

class HandGestureKeyboard:
    """Main application class with MediaPipe integration"""
    
    def __init__(self) -> None:
        print("[App] Initializing Hand Gesture Virtual Keyboard v3.0...")
        
        # Initialize application state
        self.state = AppState()
        
        # Initialize MediaPipe hand tracker
        try:
            self.hand_tracker = MediaPipeHandTracker()
            print("[App] MediaPipe hand tracker initialized successfully")
        except Exception as e:
            print(f"[App] Error initializing hand tracker: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize hand tracker:\n{e}")
            raise
        
        # Initialize webcam
        self.cap: Optional[cv2.VideoCapture] = None
        self._initialize_camera()
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("MediaPipe Hand Gesture Virtual Keyboard v3.0")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.resizable(True, True)
        
        # Make window stay on top initially
        self.root.attributes('-topmost', True)
        self.root.after(3000, lambda: self.root.attributes('-topmost', False))
        
        # GUI components
        self.key_buttons: Dict[str, tk.Button] = {}
        self.highlighted_key: Optional[tk.Button] = None
        self.current_highlighted_key_name: Optional[str] = None
        
        # Threading components
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.pinch_queue: queue.Queue[bool] = queue.Queue(maxsize=16)
        self.camera_thread: Optional[threading.Thread] = None
        
        self.setup_gui()
        self.start_camera_thread()
        
        print("[App] Application initialization complete")

    def _initialize_camera(self) -> None:
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try different camera indices
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                else:
                    raise RuntimeError("Could not open any camera")
                
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            print("[App] Camera initialized successfully")
            
            # Test camera read
            ret, test_frame = self.cap.read()
            if not ret:
                raise RuntimeError("Camera read test failed")
                
        except Exception as e:
            print(f"[App] Camera initialization failed: {e}")
            messagebox.showerror("Camera Error", 
                               f"Failed to initialize camera:\n{e}\n\nPlease check camera connection.")
            raise

    def setup_gui(self) -> None:
        """Setup the modern GUI layout"""
        # Main title
        title_label = tk.Label(
            self.root,
            text="🤖 MediaPipe Hand Tracking Virtual Keyboard v3.0",
            font=('Arial', 18, 'bold'),
            bg=BACKGROUND_COLOR,
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="👉 Right Hand: Index finger controls cursor | 👈 Left Hand: Pinch (thumb+index) to type",
            font=('Arial', 12),
            bg=BACKGROUND_COLOR,
            fg='yellow'
        )
        instructions.pack(pady=8)
        
        # Main container with proper layout
        main_container = tk.Frame(self.root, bg=BACKGROUND_COLOR)
        main_container.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Left side - Camera feed
        camera_container = tk.LabelFrame(
            main_container, 
            text="📹 Camera Feed", 
            bg=BACKGROUND_COLOR, 
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='solid',
            bd=2
        )
        camera_container.pack(side='left', padx=(0, 15), fill='both', expand=True)
        
        self.camera_label = tk.Label(
            camera_container,
            bg='black',
            text="🔄 Initializing Camera...\n\n📋 Position both hands in view:\n• Right hand for cursor\n• Left hand for clicking",
            fg='white',
            font=('Arial', 12),
            justify='center'
        )
        self.camera_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right side - Keyboard and controls
        controls_container = tk.Frame(main_container, bg=BACKGROUND_COLOR)
        controls_container.pack(side='right', fill='both', expand=True)
        
        # Status display with enhanced info
        status_frame = tk.LabelFrame(
            controls_container,
            text="📊 Status",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="🚀 Status: Starting MediaPipe...",
            font=STATUS_FONT,
            bg=BACKGROUND_COLOR,
            fg=STATUS_COLOR,
            justify='left',
            anchor='w'
        )
        self.status_label.pack(pady=8, padx=10, fill='x')
        
        # Text area for typed content
        text_frame = tk.LabelFrame(
            controls_container,
            text="📝 Typed Text",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        text_frame.pack(fill='x', pady=(0, 10))
        
        self.text_area = tk.Text(
            text_frame,
            font=TEXT_FONT,
            bg=TEXT_AREA_COLOR,
            fg='black',
            wrap='word',
            height=4,
            padx=10,
            pady=8,
            relief='sunken',
            bd=2
        )
        self.text_area.pack(fill='x', padx=8, pady=8)
        
        # Mode and statistics info
        info_frame = tk.Frame(controls_container, bg=BACKGROUND_COLOR)
        info_frame.pack(fill='x', pady=(0, 10))
        
        # Current mode display
        mode_frame = tk.Frame(info_frame, bg=BACKGROUND_COLOR)
        mode_frame.pack(side='left', fill='x', expand=True)
        
        self.mode_label = tk.Label(
            mode_frame,
            text=f"🔤 Mode: {self.state.current_mode.title()}",
            font=('Arial', 11, 'bold'),
            bg=BACKGROUND_COLOR,
            fg='cyan'
        )
        self.mode_label.pack(anchor='w')
        
        # Statistics display
        stats_frame = tk.Frame(info_frame, bg=BACKGROUND_COLOR)
        stats_frame.pack(side='right')
        
        self.stats_label = tk.Label(
            stats_frame,
            text="⌨️ Keys: 0 | 🎯 Pinches: 0 | ⏱️ Time: 0s",
            font=('Arial', 10),
            bg=BACKGROUND_COLOR,
            fg=STATUS_COLOR
        )
        self.stats_label.pack(anchor='e')
        
        # Cursor position display
        self.cursor_info_label = tk.Label(
            controls_container,
            text="🎯 Cursor: (0, 0) | 🔍 Highlighted: None",
            font=('Arial', 9),
            bg=BACKGROUND_COLOR,
            fg='lightgray'
        )
        self.cursor_info_label.pack(pady=(0, 10))
        
        # Virtual keyboard frame
        keyboard_frame = tk.LabelFrame(
            controls_container,
            text="⌨️ Virtual Keyboard",
            bg=BACKGROUND_COLOR,
            fg='white',
            font=('Arial', 11, 'bold')
        )
        keyboard_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.keyboard_frame = tk.Frame(keyboard_frame, bg=BACKGROUND_COLOR)
        self.keyboard_frame.pack(pady=10)
        
        self.create_keyboard()
        
        # Control buttons
        control_frame = tk.Frame(controls_container, bg=BACKGROUND_COLOR)
        control_frame.pack(fill='x')
        
        # Control buttons with icons
        control_buttons = [
            ("🔄 Reset", self.reset_application, '#4CAF50'),
            ("💾 Save Text", self.save_text, '#2196F3'),
            ("🗑️ Clear All", self.clear_all_text, '#FF5722'),
            ("❌ Exit", self.on_closing, '#F44336')
        ]
        
        for text, command, color in control_buttons:
            btn = tk.Button(
                control_frame,
                text=text,
                command=command,
                font=('Arial', 9, 'bold'),
                bg=color,
                fg='white',
                relief='raised',
                bd=2,
                padx=10
            )
            btn.pack(side='left', padx=3, pady=5)
        
        # Start periodic GUI updates
        self.update_display()

        # Visual cursor dot overlay (for debugging/aiming)
        try:
            self.cursor_dot = tk.Frame(self.root, bg=CURSOR_DISPLAY_COLOR, width=10, height=10)
            self.cursor_dot.place(x=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2)
        except Exception:
            self.cursor_dot = None

    def create_keyboard(self) -> None:
        """Create the virtual keyboard with enhanced styling"""
        # Clear existing keyboard
        for widget in self.keyboard_frame.winfo_children():
            widget.destroy()
        
        self.key_buttons.clear()
        layout = KEYBOARD_LAYOUTS[self.state.current_mode]
        
        for row_idx, row in enumerate(layout):
            row_frame = tk.Frame(self.keyboard_frame, bg=BACKGROUND_COLOR)
            row_frame.pack(pady=3)
            
            for key in row:
                # Determine button properties based on key type
                if key == 'SPACE':
                    width, text, bg_color = 15, '⎵ SPACE', KEY_COLOR
                elif key == 'BACKSPACE':
                    width, text, bg_color = 10, '⌫ BACK', '#FF6B6B'
                elif key == 'CLEAR':
                    width, text, bg_color = 8, '🗑️ CLR', '#FF8A65'
                elif key == 'CAPS':
                    width, text, bg_color = 8, '🔄 CAPS', '#9C27B0'
                elif key == 'NUMBERS':
                    width, text, bg_color = 8, '🔢 NUM', '#673AB7'
                elif key == 'LETTERS':
                    width, text, bg_color = 8, '🔤 ABC', '#3F51B5'
                else:
                    width, text, bg_color = 4, key, KEY_COLOR
                
                btn = tk.Button(
                    row_frame,
                    text=text,
                    width=width,
                    height=2,
                    font=KEY_FONT,
                    bg=bg_color,
                    fg='white',
                    relief='raised',
                    bd=2,
                    activebackground=KEY_HOVER_COLOR,
                    cursor='hand2'
                )
                btn.pack(side='left', padx=2, pady=1)
                self.key_buttons[key] = btn

    def start_camera_thread(self) -> None:
        """Start camera processing in separate thread"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        print("[App] Camera thread started")

    def camera_loop(self) -> None:
        """Main camera processing loop with MediaPipe integration"""
        print("[App] Camera processing loop started")
        frame_count = 0
        last_log_time = time.time()
        
        while self.state.running and self.cap is not None:
            try:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("[App] Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Process frame with MediaPipe hand tracker
                processed_frame, pinch_detected = self.hand_tracker.process_frame(frame)
                
                # Handle pinch detection for key pressing (defer to GUI thread)
                if pinch_detected:
                    try:
                        if not self.pinch_queue.full():
                            self.pinch_queue.put_nowait(True)
                    except Exception:
                        pass
                
                # Queue processed frame for GUI display (non-blocking)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                
                frame_count += 1
                
                # Periodic logging
                current_time = time.time()
                if current_time - last_log_time >= 30.0:  # Log every 30 seconds
                    print(f"[App] Processed {frame_count} frames in camera loop")
                    last_log_time = current_time
                    frame_count = 0
                
            except Exception as e:
                print(f"[App] Error in camera loop: {e}")
                time.sleep(0.1)  # Brief pause on error
            
            time.sleep(1/CAMERA_FPS)  # Maintain target framerate

    def handle_pinch_detected(self) -> None:
        """Handle pinch detection from left hand"""
        if self.highlighted_key is None or self.current_highlighted_key_name is None:
            print("[App] Pinch detected but no key is highlighted")
            return
        
        key_name = self.current_highlighted_key_name
        self.execute_key_action(key_name)
        
        # Visual feedback for successful pinch
        self.highlighted_key.config(bg=KEY_PRESSED_COLOR)
        self.root.after(300, lambda: self.reset_key_color(self.highlighted_key))
        
        # Update pinch counter
        self.state.pinch_count += 1
        
        print(f"[App] Pinch executed key: '{key_name}' at {time.strftime('%H:%M:%S')}")

    def execute_key_action(self, key_name: str) -> None:
        """Execute the action for a given key"""
        if key_name == 'SPACE':
            self.state.typed_text += ' '
            self.state.key_count += 1
        elif key_name == 'BACKSPACE':
            if self.state.typed_text:
                self.state.typed_text = self.state.typed_text[:-1]
            self.state.key_count += 1
        elif key_name == 'CLEAR':
            self.state.typed_text = ""
            self.state.key_count += 1
        elif key_name == 'CAPS':
            new_mode = 'uppercase' if self.state.current_mode == 'lowercase' else 'lowercase'
            self.switch_keyboard_mode(new_mode)
        elif key_name == 'NUMBERS':
            self.switch_keyboard_mode('numbers')
        elif key_name == 'LETTERS':
            self.switch_keyboard_mode('lowercase')
        else:
            # Regular character key
            self.state.typed_text += key_name
            self.state.key_count += 1
        
        # Update text display
        self.update_text_display()

    def switch_keyboard_mode(self, new_mode: str) -> None:
        """Switch keyboard mode and recreate keyboard"""
        if new_mode in KEYBOARD_LAYOUTS and new_mode != self.state.current_mode:
            self.state.current_mode = new_mode
            self.mode_label.config(text=f"🔤 Mode: {self.state.current_mode.title()}")
            self.create_keyboard()
            print(f"[App] Switched to {new_mode} keyboard mode")

    def update_display(self) -> None:
        """Main GUI update loop"""
        try:
            # Drain pinch events and execute on GUI thread
            while not self.pinch_queue.empty():
                try:
                    _ = self.pinch_queue.get_nowait()
                    self.handle_pinch_detected()
                except Exception:
                    break
            # Update camera feed display
            try:
                frame = self.frame_queue.get_nowait()
                self.display_camera_feed(frame)
            except queue.Empty:
                pass  # No new frame available
            
            # Update cursor highlighting on keyboard
            self.update_cursor_highlighting()
            
            # Update status information
            self.update_status_display()
            
            # Update statistics
            self.update_statistics_display()
            
        except Exception as e:
            print(f"[App] Error in display update: {e}")
        
        # Schedule next update
        if self.state.running:
            self.root.after(33, self.update_display)  # ~30 FPS GUI updates

    def display_camera_feed(self, frame: np.ndarray) -> None:
        """Display camera feed in GUI with proper scaling"""
        try:
            # Resize frame to fit display area while maintaining aspect ratio
            display_height = 400
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            display_width = int(display_height * aspect_ratio)
            
            # Resize frame
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update camera label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep reference to prevent garbage collection
            
        except Exception as e:
            print(f"[App] Error displaying camera feed: {e}")
            self.camera_label.config(
                text="📷 Camera Display Error\n\nCheck camera connection\nand restart application",
                image=""
            )

    def update_cursor_highlighting(self) -> None:
        """Update keyboard key highlighting based on cursor position"""
        cursor_x, cursor_y = self.hand_tracker.get_cursor_position()

        # Cursor from tracker is already in root window coordinates; no extra remap

        # Clamp to root client area
        try:
            root_w = max(1, self.root.winfo_width())
            root_h = max(1, self.root.winfo_height())
            cursor_x = max(0, min(root_w - 1, cursor_x))
            cursor_y = max(0, min(root_h - 1, cursor_y))
        except Exception:
            pass

        # Move the visible cursor dot overlay
        if getattr(self, 'cursor_dot', None) is not None:
            try:
                self.cursor_dot.place(x=cursor_x - 5, y=cursor_y - 5)
            except Exception:
                pass
        
        # Reset all key colors to default
        for key_name, btn in self.key_buttons.items():
            if btn != self.highlighted_key:
                # Set default color based on key type
                if key_name == 'BACKSPACE':
                    btn.config(bg='#FF6B6B')
                elif key_name == 'CLEAR':
                    btn.config(bg='#FF8A65')
                elif key_name == 'CAPS':
                    btn.config(bg='#9C27B0')
                elif key_name in ['NUMBERS', 'LETTERS']:
                    btn.config(bg='#673AB7')
                else:
                    btn.config(bg=KEY_COLOR)
        
        # Find key under cursor
        new_highlighted_key = None
        new_highlighted_key_name = None
        
        for key_name, btn in self.key_buttons.items():
            if btn.winfo_viewable():
                try:
                    # Get button absolute position relative to root window
                    self.root.update_idletasks()
                    btn_x = btn.winfo_rootx() - self.root.winfo_rootx()
                    btn_y = btn.winfo_rooty() - self.root.winfo_rooty()
                    btn_w = btn.winfo_width()
                    btn_h = btn.winfo_height()
                    
                    # Check if cursor is within button bounds
                    if (btn_x <= cursor_x <= btn_x + btn_w and 
                        btn_y <= cursor_y <= btn_y + btn_h):
                        new_highlighted_key = btn
                        new_highlighted_key_name = key_name
                        btn.config(bg=KEY_HOVER_COLOR)
                        break
                        
                except tk.TclError:
                    # Button may not be fully initialized yet
                    pass
        
        # Update highlighted key references
        self.highlighted_key = new_highlighted_key
        self.current_highlighted_key_name = new_highlighted_key_name
        
        # Update cursor info display
        highlighted_text = new_highlighted_key_name if new_highlighted_key_name else "None"
        self.cursor_info_label.config(
            text=f"🎯 Cursor: ({cursor_x}, {cursor_y}) | 🔍 Highlighted: {highlighted_text}"
        )

    def update_status_display(self) -> None:
        """Update comprehensive status display"""
        right_detected = self.hand_tracker.is_right_hand_detected()
        left_detected = self.hand_tracker.is_left_hand_detected()
        
        if right_detected and left_detected:
            status = "✅ Both hands detected - Ready to type!"
            color = STATUS_COLOR
        elif right_detected:
            status = "⚠️ Right hand detected - Show left hand to click"
            color = 'orange'
        elif left_detected:
            status = "⚠️ Left hand detected - Show right hand for cursor"
            color = 'orange'
        else:
            status = "❌ No hands detected - Show both hands to camera"
            color = 'red'
        
        # Add detailed status
        detection_status = self.hand_tracker.get_detection_status()
        full_status = f"{status}\n{detection_status}"
        
        self.status_label.config(text=full_status, fg=color)

    def update_statistics_display(self) -> None:
        """Update statistics display with comprehensive metrics"""
        elapsed_time = int(time.time() - self.state.start_time)
        
        # Calculate typing speed (characters per minute)
        cpm = int(len(self.state.typed_text) / max(elapsed_time, 1) * 60) if elapsed_time > 0 else 0
        
        # Update statistics
        stats_text = (f"⌨️ Keys: {self.state.key_count} | "
                     f"🎯 Pinches: {self.state.pinch_count} | "
                     f"⏱️ Time: {elapsed_time}s | "
                     f"🚀 CPM: {cpm}")
        
        self.stats_label.config(text=stats_text)

    def update_text_display(self) -> None:
        """Update the text display area"""
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, self.state.typed_text)
        self.text_area.see(tk.END)  # Auto-scroll to end

    def reset_key_color(self, btn: Optional[tk.Button]) -> None:
        """Reset key color after press animation"""
        if btn and btn in self.key_buttons.values():
            # Find the key name to set appropriate default color
            key_name = None
            for name, button in self.key_buttons.items():
                if button == btn:
                    key_name = name
                    break
            
            if key_name:
                if key_name == 'BACKSPACE':
                    btn.config(bg='#FF6B6B')
                elif key_name == 'CLEAR':
                    btn.config(bg='#FF8A65')
                elif key_name == 'CAPS':
                    btn.config(bg='#9C27B0')
                elif key_name in ['NUMBERS', 'LETTERS']:
                    btn.config(bg='#673AB7')
                else:
                    btn.config(bg=KEY_COLOR)

    def reset_application(self) -> None:
        """Reset application to initial state"""
        self.state.typed_text = ""
        self.state.key_count = 0
        self.state.pinch_count = 0
        self.state.start_time = time.time()
        self.state.current_mode = 'lowercase'
        
        self.create_keyboard()
        self.update_text_display()
        self.mode_label.config(text=f"🔤 Mode: {self.state.current_mode.title()}")
        
        print("[App] Application reset to initial state")

    def save_text(self) -> None:
        """Save typed text to file"""
        if not self.state.typed_text.strip():
            messagebox.showwarning("Save Text", "No text to save!")
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Typed Text"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.state.typed_text)
                messagebox.showinfo("Save Text", f"Text saved successfully to:\n{filename}")
                print(f"[App] Text saved to: {filename}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save text:\n{e}")
            print(f"[App] Save error: {e}")

    def clear_all_text(self) -> None:
        """Clear all typed text with confirmation"""
        if self.state.typed_text.strip():
            result = messagebox.askyesno(
                "Clear Text", 
                "Are you sure you want to clear all typed text?"
            )
            if result:
                self.state.typed_text = ""
                self.update_text_display()
                print("[App] All text cleared")
        else:
            messagebox.showinfo("Clear Text", "No text to clear!")

    def run(self) -> None:
        """Run the application with comprehensive error handling"""
        try:
            # Set up window close protocol
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            print("[App] Starting GUI main loop...")
            print("[App] Application ready for hand tracking!")
            
            # Start the main GUI loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("[App] Application interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"[App] Unexpected error in main loop: {e}")
            messagebox.showerror("Application Error", f"Unexpected error:\n{e}")
            self.on_closing()

    def on_closing(self) -> None:
        """Handle application closing with proper cleanup"""
        print("[App] Shutting down application...")
        
        try:
            # Stop the main loop
            self.state.running = False
            
            # Give threads time to finish
            time.sleep(0.5)
            
            # Cleanup camera
            if self.cap is not None:
                self.cap.release()
                print("[App] Camera released")
            
            # Cleanup hand tracker (MediaPipe resources)
            if hasattr(self, 'hand_tracker'):
                self.hand_tracker.cleanup()
                print("[App] Hand tracker cleaned up")
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            
            # Destroy GUI
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
            
        except Exception as e:
            print(f"[App] Error during cleanup: {e}")
        
        print("[App] Application shutdown complete")
        sys.exit(0)

def main():
    """Main entry point with comprehensive startup checks"""
    print("=" * 60)
    print("🚀 MediaPipe Hand Gesture Virtual Keyboard v3.0")
    print("=" * 60)
    print("📋 Instructions:")
    print("   • 👉 Right Hand: Move index finger to control cursor")
    print("   • 👈 Left Hand: Pinch thumb + index finger to click keys")
    print("   • 👀 Both hands must be visible to camera")
    print("   • 📹 Ensure webcam is connected and working")
    print("   • 🔄 Green lines and red dots will appear on detected hands")
    print(f"   • 🐍 Running on Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: This application requires Python 3.8 or higher")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    elif sys.version_info >= (3, 13):
        print("✅ Excellent! Running on Python 3.13+ with full compatibility")
    
    # Check required dependencies (MediaPipe optional with OpenCV fallback)
    missing: list[str] = []
    try:
        import cv2 as _cv2  # noqa: F401
    except Exception:
        missing.append("opencv-python")
    try:
        import PIL  # noqa: F401
    except Exception:
        missing.append("pillow")
    try:
        import mediapipe  # noqa: F401
        has_mp = True
    except Exception:
        has_mp = False
    if missing:
        print(f"❌ Missing required dependency(s): {', '.join(missing)}")
        print("📦 Please install required packages:")
        print("   pip install " + " ".join(missing))
        sys.exit(1)
    if not has_mp:
        print("⚠️ MediaPipe not found. Running with OpenCV-only fallback (no tracking).")
        print("   To enable full tracking later, install mediapipe when Py3.13 wheels are available.")
    
    # Start application
    try:
        app = HandGestureKeyboard()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check camera connection and permissions")
        print("   • Ensure no other applications are using the camera")
        print("   • Try running as administrator if on Windows")
        print("   • Check that MediaPipe is properly installed")
        sys.exit(1)

if __name__ == "__main__":
    main()