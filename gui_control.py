import time
import cv2
import torch
import websocket
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sounddevice as sd
from ultralytics import YOLO
from transformers import pipeline

# ==================== SYSTEM CONFIGURATION ====================
ESP32_IP = "10.211.65.189"          
CAMERA_STREAM_URL = "http://10.211.65.247:8080/video"  

SAMPLE_RATE = 16000                  
RECORD_DURATION = 4                  

FRAME_WIDTH = 640
FRAME_CENTER = 320                   
MIN_SPEED = 50                       
# ==============================================================

class FreshVideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret; self.frame = frame  

    def read(self): return self.ret, self.frame
    def release(self): self.running = False; self.cap.release()


class RobotControlGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("RTX 3050 - Robot Ground Control Station")
        self.window.geometry("1050x800")
        self.window.configure(bg="#1e1e2e")

        self.target_keyword = "chair"
        self.is_recording = False
        
        # DYNAMIC CONTROL STATES
        self.nav_enabled = False  
        self.base_speed = 90      
        self.kp = 0.15            
        self.scan_speed = 80      

        print("Loading AI Models...")
        self.yolo_model = YOLO("yolov8n.pt").to("cuda")
        self.whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)
        
        print("Connecting to Robot...")
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"ws://{ESP32_IP}:81")
        except Exception:
            print("Warning: ESP32 offline.")
            self.ws = None

        self.cap = FreshVideoStream(CAMERA_STREAM_URL)
        self.setup_ui()
        self.update_gui_loop()

    def setup_ui(self):
        # --- Left Side: Video Output Canvas ---
        self.video_label = tk.Label(self.window, bg="#000000")
        self.video_label.pack(side=tk.LEFT, padx=20, pady=20)

        # --- Right Side: Dashboard Control Tower ---
        self.panel = tk.Frame(self.window, bg="#252538", padx=15, pady=15)
        self.panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=20)

        # HUD Title
        tk.Label(self.panel, text="GROUND CONTROL SYSTEM", font=("Arial", 14, "bold"), fg="#cdd6f4", bg="#252538").pack(pady=10)

        # Mode Indicator Status
        self.status_label = tk.Label(self.panel, text="SYSTEM STATUS: MANUAL OVERRIDE", font=("Arial", 11, "bold"), fg="#fab387", bg="#252538")
        self.status_label.pack(pady=5)

        self.target_label = tk.Label(self.panel, text=f"AI TARGET: {self.target_keyword.upper()}", font=("Arial", 11, "bold"), fg="#bac2de", bg="#252538")
        self.target_label.pack(pady=5)

        # --- SLIDER CONTROLS CONFIGURATION ---
        sliders_frame = tk.LabelFrame(self.panel, text=" MOTOR CALIBRATION ", font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#252538", bd=1, padx=10, pady=10)
        sliders_frame.pack(fill=tk.X, pady=10)

        # 1. Forward Cruise Speed Slider
        tk.Label(sliders_frame, text="FORWARD CRUISE SPEED", font=("Arial", 9, "bold"), fg="#bac2de", bg="#252538").pack(anchor=tk.W)
        self.speed_slider = tk.Scale(sliders_frame, from_=60, to=200, orient=tk.HORIZONTAL, bg="#252538", fg="#cdd6f4", highlightthickness=0, command=self.update_speed_value)
        self.speed_slider.set(self.base_speed)
        self.speed_slider.pack(fill=tk.X, pady=(0, 10))

        # 2. Dynamic Turn Speed (KP) Sensitivity Slider
        tk.Label(sliders_frame, text="TURN SENSITIVITY (KP)", font=("Arial", 9, "bold"), fg="#bac2de", bg="#252538").pack(anchor=tk.W)
        self.turn_slider = tk.Scale(sliders_frame, from_=0.05, to=0.50, resolution=0.01, orient=tk.HORIZONTAL, bg="#252538", fg="#cdd6f4", highlightthickness=0, command=self.update_turn_value)
        self.turn_slider.set(self.kp)
        self.turn_slider.pack(fill=tk.X, pady=(0, 10))

        # 3. Dedicated Scan Rotation Speed Slider
        tk.Label(sliders_frame, text="ENVIRONMENT SCAN SPEED", font=("Arial", 9, "bold"), fg="#bac2de", bg="#252538").pack(anchor=tk.W)
        self.scan_slider = tk.Scale(sliders_frame, from_=50, to=150, orient=tk.HORIZONTAL, bg="#252538", fg="#cdd6f4", highlightthickness=0, command=self.update_scan_value)
        self.scan_slider.set(self.scan_speed)
        self.scan_slider.pack(fill=tk.X)

        # Operation Action Triggers
        self.mic_button = tk.Button(self.panel, text="🎤 Voice Target Command", font=("Arial", 10, "bold"), bg="#fab387", fg="#11111b", command=self.start_voice_thread, height=2)
        self.mic_button.pack(fill=tk.X, pady=5)

        self.ai_toggle_button = tk.Button(self.panel, text="🚀 ENGAGE AI TRACKING", font=("Arial", 10, "bold"), bg="#a6e3a1", fg="#11111b", command=lambda: self.toggle_ai_navigation(True), height=2)
        self.ai_toggle_button.pack(fill=tk.X, pady=5)

        # --- SAFETY ACTION ROW (SIDE-BY-SIDE PLACEMENT) ---
        safety_row_frame = tk.Frame(self.panel, bg="#252538")
        safety_row_frame.pack(fill=tk.X, pady=5)

        # Left Column: Emergency Lock
        self.stop_button = tk.Button(safety_row_frame, text="🛑 EMERGENCY LOCK", font=("Arial", 10, "bold"), bg="#f38ba8", fg="#11111b", command=self.emergency_lock, height=2, width=20)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Right Column: Exit Mission & Script Stop
        self.exit_button = tk.Button(safety_row_frame, text="❌ EXIT MISSION", font=("Arial", 10, "bold"), bg="#585b70", fg="#cdd6f4", activebackground="#f38ba8", command=self.cleanup, height=2, width=20)
        self.exit_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # --- MANUAL DRIVING INTERFACE CLUSTER ---
        drive_frame = tk.LabelFrame(self.panel, text=" MANUAL DRIVE CONTROLS ", font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#252538", bd=1, padx=10, pady=10)
        drive_frame.pack(fill=tk.X, pady=10)
        
        grid_frame = tk.Frame(drive_frame, bg="#252538")
        grid_frame.pack()

        btn_w = tk.Button(grid_frame, text="▲ FORWARD", font=("Arial", 9, "bold"), bg="#89b4fa", fg="#11111b", command=lambda: self.manual_drive("F"), width=12, height=2)
        btn_a = tk.Button(grid_frame, text="◀ LEFT", font=("Arial", 9, "bold"), bg="#89b4fa", fg="#11111b", command=lambda: self.manual_drive("L"), width=9, height=2)
        btn_s = tk.Button(grid_frame, text="▼ REVERSE", font=("Arial", 9, "bold"), bg="#89b4fa", fg="#11111b", command=lambda: self.manual_drive("B"), width=12, height=2)
        btn_d = tk.Button(grid_frame, text="▶ RIGHT", font=("Arial", 9, "bold"), bg="#89b4fa", fg="#11111b", command=lambda: self.manual_drive("R"), width=9, height=2)
        btn_stop = tk.Button(grid_frame, text="🛑 BRAKE", font=("Arial", 9, "bold"), bg="#f38ba8", fg="#11111b", command=self.send_stop_packet, width=12, height=2)

        btn_w.grid(row=0, column=1, pady=3)
        btn_a.grid(row=1, column=0, padx=3)
        btn_stop.grid(row=1, column=1) 
        btn_d.grid(row=1, column=2, padx=3)
        btn_s.grid(row=2, column=1, pady=3)

    def update_speed_value(self, val):
        self.base_speed = int(val)

    def update_turn_value(self, val):
        self.kp = float(val)

    def update_scan_value(self, val):
        self.scan_speed = int(val)

    def toggle_ai_navigation(self, target_state):
        if target_state and not self.nav_enabled:
            self.nav_enabled = True
            self.status_label.config(text="SYSTEM STATUS: AI AUTONOMOUS", fg="#a6e3a1")
            self.ai_toggle_button.config(text="⏸ PAUSE AI MODE", bg="#f9e2af")
        else:
            self.nav_enabled = False
            self.status_label.config(text="SYSTEM STATUS: AI PAUSED / MANUAL", fg="#fab387")
            self.ai_toggle_button.config(text="🚀 ENGAGE AI TRACKING", bg="#a6e3a1")
            self.send_stop_packet()

    def manual_drive(self, direction):
        if self.nav_enabled:
            self.toggle_ai_navigation(False)
        
        if not self.ws: return
        
        speed = self.base_speed
        if direction == "F": self.ws.send(f"M,{speed},{speed}")
        elif direction == "B": self.ws.send(f"M,{-speed},{-speed}")
        elif direction == "L": self.ws.send(f"M,{-speed},{speed}")  
        elif direction == "R": self.ws.send(f"M,{speed},{-speed}")

    def send_stop_packet(self):
        if self.ws: self.ws.send("S")

    def emergency_lock(self):
        print("🚨 EMERGENCY SHIELD LOCK ENGAGED.")
        self.nav_enabled = False
        self.status_label.config(text="SYSTEM STATUS: LOCKED / EMERGENCY STOP", fg="#f38ba8")
        self.ai_toggle_button.config(text="🚀 ENGAGE AI TRACKING", bg="#a6e3a1")
        self.send_stop_packet()

    def start_voice_thread(self):
        if not self.is_recording:
            threading.Thread(target=self.handle_voice_command, daemon=True).start()

    def handle_voice_command(self):
        self.is_recording = True
        self.mic_button.config(text="🎙️ Listening...", bg="#f38ba8")
        
        audio_data = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        
        self.mic_button.config(text="⚙️ Transcribing...", bg="#f9e2af")
        audio_array = audio_data.flatten()
        
        try:
            text_command = self.whisper_asr({"raw": audio_array, "sampling_rate": SAMPLE_RATE})['text'].lower()
            for obj in ["chair", "bottle", "cup", "person", "laptop", "backpack", "book"]:
                if obj in text_command:
                    self.target_keyword = obj
                    self.target_label.config(text=f"AI TARGET: {self.target_keyword.upper()}")
                    break
        except Exception as e:
            print(f"Voice failure: {e}")
            
        self.mic_button.config(text="🎤 Voice Target Command", bg="#fab387")
        self.is_recording = False

    def update_gui_loop(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            results = self.yolo_model(frame, verbose=False, half=True, imgsz=320)
            target_found = False

            for result in results:
                for box in result.boxes:
                    label = self.yolo_model.names[int(box.cls[0])]
                    if label == self.target_keyword:
                        target_found = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_width = x2 - x1
                        delta_x = ((x1 + x2) / 2) - FRAME_CENTER
                        proximity_ratio = box_width / FRAME_WIDTH

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {proximity_ratio:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if self.nav_enabled:
                            if proximity_ratio > 0.85:
                                print("🏁 Target reached.")
                                self.emergency_lock()
                            else:
                                steering_bias = int(delta_x * self.kp)
                                left_speed = max(MIN_SPEED, min(self.base_speed + steering_bias, 230))
                                right_speed = max(MIN_SPEED, min(self.base_speed - steering_bias, 230))
                                if self.ws: self.ws.send(f"M,{left_speed},{right_speed}")
                        break
                if target_found: break

            if not target_found and self.nav_enabled:
                if self.ws: self.ws.send(f"M,{-self.scan_speed},{self.scan_speed}")

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.imgtk = tk_image
            self.video_label.configure(image=tk_image)

        self.window.after(10, self.update_gui_loop)

    def cleanup(self):
        print("\n[Shutting Down] Terminating ground control structures cleanly...")
        self.nav_enabled = False
        self.send_stop_packet()
        if self.ws: 
            try: self.ws.close()
            except Exception: pass
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()