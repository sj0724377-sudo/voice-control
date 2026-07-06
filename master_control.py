import time
import cv2
import torch
import websocket
import numpy as np
import threading
import sounddevice as sd
from ultralytics import YOLO
from transformers import pipeline

# ==================== 1. SYSTEM CONFIGURATION ====================
ESP32_IP = "10.211.65.189"          
CAMERA_STREAM_URL = "http://10.211.65.247:8080/video"  

SAMPLE_RATE = 16000                  
RECORD_DURATION = 4                  

# HIGH-PERFORMANCE CONTINUOUS STREAMING CONFIGURATION
FRAME_WIDTH = 640
FRAME_CENTER = 320                   

# Cruise Speed settings (0 to 255 PWM)
BASE_SPEED = 70                   # Normal cruise speed for straight-line tracking
MIN_SPEED = 60                      # The floor threshold to prevent stalling out entirely
KP = 0.35                          # Steering sensitivity adjustment multiplier
# =================================================================

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

# Initialize AI
print("\nBooting real-time continuous tracker...")
yolo_model = YOLO("yolov8n.pt").to("cuda")
whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)

# Connect to Wireless Tunnel
ws = websocket.WebSocket()
ws.connect(f"ws://{ESP32_IP}:81")

# Voice command parsing
print("\n🎤 Listening for tracking target command...")
audio_data = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio_array = audio_data.flatten()
try:
    text_command = whisper_asr({"raw": audio_array, "sampling_rate": SAMPLE_RATE})['text'].lower()
except Exception:
    text_command = ""

target_keyword = "chair"
for obj in ["chair", "bottle", "cup", "person", "laptop", "backpack", "book"]:
    if obj in text_command: target_keyword = obj; break
print(f"🎯 Mission objective focused on: [{target_keyword.upper()}]\n")

cap = FreshVideoStream(CAMERA_STREAM_URL)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue

        results = yolo_model(frame, verbose=False, half=True, imgsz=320)
        target_found = False

        for result in results:
            for box in result.boxes:
                if yolo_model.names[int(box.cls[0])] == target_keyword:
                    target_found = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate tracking data metrics
                    box_width = x2 - x1
                    delta_x = ((x1 + x2) / 2) - FRAME_CENTER
                    proximity_ratio = box_width / FRAME_WIDTH

                    # Render Diagnostics
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 1. Arrival Check
                    if proximity_ratio > 0.85:
                        print("🏁 Target Reached! Braking.")
                        ws.send("S")
                        time.sleep(1.0)
                    
                    # 2. Continuous Proportional Steering Vector Generation
                    else:
                        steering_bias = int(delta_x * KP)
                        
                        # Calculate fluid differential speeds
                        left_speed = BASE_SPEED + steering_bias
                        right_speed = BASE_SPEED - steering_bias

                        # Constrain inside real hardware safety limits
                        left_speed = max(MIN_SPEED, min(left_speed, 230))
                        right_speed = max(MIN_SPEED, min(right_speed, 230))

                        print(f"🎯 Error: {int(delta_x)}px | Left PWM: {left_speed} | Right PWM: {right_speed}")
                        ws.send(f"M,{left_speed},{right_speed}")
                    
                    break
            if target_found: break

        # If target leaves frame, enter a continuous searching rotation scan
        if not target_found:
            print("🔍 Target missing. Continuous scan sweeping...")
            ws.send(f"M,{-BASE_SPEED},{BASE_SPEED}") # Low-speed zero-radius rotation search

        cv2.imshow("RTX 3050 - Continuous Tracking Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nShutting down.")
finally:
    ws.send("S"); ws.close(); cap.release(); cv2.destroyAllWindows()