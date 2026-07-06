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
ESP32_IP = "10.211.65.189"          # <--- Hand-matched to your active ESP32 IP 
CAMERA_STREAM_URL = "http://10.211.65.247:8080/video"  # <--- Hand-matched to your phone stream IP

# Audio Capture Settings
SAMPLE_RATE = 16000                  
RECORD_DURATION = 4                  

# Stop-and-Scan Tuning Thresholds
PULSE_DURATION = 0.08                
STREAM_DELAY = 0.15                  
FRAME_WIDTH = 640
FRAME_CENTER = 320                   
DEADZONE = 85                        
# =================================================================



# ==================== 2. LOW-LATENCY VIDEO STREAM THREAD ====================
class FreshVideoStream:
    """
    Spawns a background thread to continuously dump OpenCV's internal frame queue.
    This bypasses network lag, ensuring python always processes the absolute newest frame.
    """
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set internal driver buffer size to minimum
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
                self.ret = ret
                self.frame = frame  # Instantly overwrite with the fresh incoming array

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()


# ==================== 3. INITIALIZE GPU AI PIPELINES ====================
print("\n==================================================")
print("  Initializing AI Engines on NVIDIA RTX 3050...  ")
print("==================================================")

if not torch.cuda.is_available():
    print("❌ Critical Error: CUDA is not available. Check your PyTorch environment configuration.")
    exit()

print(f"CUDA Version: {torch.version.cuda} | Hardware: {torch.cuda.get_device_name(0)}")

# Load YOLOv8 directly onto GPU
print("-> Loading YOLOv8 nano visual weights...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to("cuda")

# Load Whisper Pipeline onto GPU
print("-> Initializing OpenAI Whisper automated speech recognition...")
whisper_asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0,
    generate_kwargs={"language": "en", "task": "transcribe"}
)

print("🚀 All AI Models successfully cached onto GPU cores!\n")


# ==================== 4. WIRELESS NETWORK TUNNEL SETUP ====================
WS_URL = f"ws://{ESP32_IP}:81"
print(f"Connecting to ESP32 WebSocket server at: {WS_URL}")
try:
    ws = websocket.WebSocket()
    ws.connect(WS_URL, timeout=4.0)
    print("🔌 Full-Duplex Wireless Connection Established successfully!")
except Exception as e:
    print(f"❌ Connection Failed: {e}\nVerify ESP32 power stability and hotspot status.")
    exit()


# ==================== 5. VOICE COMMAND CAPTURE & PARSING (FFMPEG BYPASS) ====================
print("\n🎤 Prepare to speak your command...")
print(f"Recording starting now! Speak clearly for the next {RECORD_DURATION} seconds...")
print("--------------------------------------------------")

audio_data = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  
print("📥 Audio recording complete. Processing raw data directly in memory...")

audio_array = audio_data.flatten()

print("🤖 Transcribing audio via Whisper...")
try:
    inference_result = whisper_asr({"raw": audio_array, "sampling_rate": SAMPLE_RATE})
    text_command = inference_result['text'].strip().lower()
    print(f"📝 Decoded Voice Command: \"{text_command}\"")
except Exception as e:
    print(f"❌ Transcription error occurred: {e}")
    text_command = ""

target_keyword = None
supported_objects = ["chair", "bottle", "cup", "person", "laptop", "backpack", "book"]

for obj in supported_objects:
    if obj in text_command:
        target_keyword = obj
        break

if target_keyword is None:
    print("⚠️ No valid target found in speech. Defaulting search focus to: 'chair'")
    target_keyword = "chair"
else:
    print(f"🎯 Target confirmed. Mission parameter set to look for: [{target_keyword.upper()}]")


# ==================== 6. LOW-LATENCY TRACKING LOOP ====================
print("\n==================================================")
print(f"   Initiating Real-Time Visual Tracking Mode for: {target_keyword.upper()}")
print("   Press 'q' on the video window or Ctrl+C to abort.  ")
print("==================================================\n")

# Connect via our low-latency custom background thread structure
cap = FreshVideoStream(CAMERA_STREAM_URL)

try:
    while True:
        # Pull the absolute newest frame captured by the background thread
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Run high-speed GPU vision inference with FP16 Half-Precision hardware optimization
        results = yolo_model(frame, verbose=False, half=True)
        target_found = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]

                if label == target_keyword:
                    target_found = True
                    
                    # Extract bounding metrics
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_width = x2 - x1
                    x_center = (x1 + x2) / 2
                    delta_x = x_center - FRAME_CENTER
                    proximity_ratio = box_width / FRAME_WIDTH

                    # Render on-screen HUD diagnostics
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {proximity_ratio:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"🎯 Locked! Delta X: {int(delta_x)} px | Width Ratio: {proximity_ratio:.2f}")

                    # --- NAVIGATION MANAGEMENT EXECUTOR ---
                    
                    # 1. Proximity Check (Stop when near the chair)
                    if proximity_ratio > 0.88:
                        print(f"🏁 Target reached ({proximity_ratio:.2f})! Applying emergency brakes.")
                        ws.send('S')
                        time.sleep(1.0) 
                        
                    # 2. Left Look-Adjustment Pulse
                    elif delta_x < -DEADZONE:
                        print("◀ Sending LEFT Pulse...")
                        ws.send('L')
                        time.sleep(PULSE_DURATION)
                        ws.send('S')  
                        time.sleep(STREAM_DELAY)  

                    # 3. Right Look-Adjustment Pulse
                    elif delta_x > DEADZONE:
                        print("▶ Sending RIGHT Pulse...")
                        ws.send('R')
                        time.sleep(PULSE_DURATION)
                        ws.send('S')  
                        time.sleep(STREAM_DELAY)  

                    # 4. Straight Line Advancement Pulse
                    else:
                        print("🚀 Centered! Advancing Forward...")
                        ws.send('F')
                        time.sleep(0.08)  # Short punch forward (80ms)
                        ws.send('S')  
                        time.sleep(STREAM_DELAY)
                    
                    break  
            if target_found:
                break

        # Controlled search rotation if target drops out of sight completely
        if not target_found:
            print("🔍 Target not in view. Scanning environment safely...")
            ws.send('L')
            time.sleep(PULSE_DURATION)
            ws.send('S')
            time.sleep(STREAM_DELAY)

        # Display window
        cv2.imshow("RTX 3050 - Autonomous Zero-Lag Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nUser manually triggered script shutdown.")

finally:
    print("\nCleaning up systems cleanly...")
    try:
        ws.send('S')
        ws.close()
        print("🔌 Robot wireless connection terminated safely.")
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("🖥️ Video capture streams released.")