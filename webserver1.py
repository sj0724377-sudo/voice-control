import os
import time
import cv2
import torch
import websocket
import threading
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from transformers import pipeline
from pydub import AudioSegment

app = Flask(__name__)

# ==================== SYSTEM CONFIGURATION ====================
ESP32_IP = "10.211.65.189"          
CAMERA_STREAM_URL = "http://10.211.65.247:8080/video"  

FRAME_WIDTH = 640
FRAME_CENTER = 320                   
MIN_SPEED = 50                       

# Global Dynamic Tracking States shared across network handlers
control_state = {
    "nav_enabled": False,
    "target_keyword": "bottle",  # Default starting target
    "base_speed": 90,
    "kp": 0.15,
    "scan_speed": 80             # Smooth continuous rotational velocity
}

# HYSTERESIS CONFIGURATION: Stabilizes physical tracking dropping frames
HYSTERESIS = {
    "max_lost_frames": 6,        # Grace period frames to ride through flicker
    "lost_frame_counter": 0,
    "last_known_delta_x": 0,
    "last_known_label": ""
}
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

print("\n========================================================")
print("  Initializing High-Precision AI Suite on NVIDIA GPU... ")
print("========================================================")

# FIX 1: Load your customized tracking brain instead of generic coco weights
yolo_model = YOLO("best.pt").to("cuda")

whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device=0
)

print("🚀 S-Tier AI Architecture successfully cached on VRAM hardware!\n")

try:
    ws = websocket.WebSocket()
    ws.connect(f"ws://{ESP32_IP}:81")
    print("Connected to ESP32 Gateway WebSocket Server.")
except Exception:
    print("Warning: ESP32 offline. Operating in local simulation mode.")
    ws = None

video_stream = FreshVideoStream(CAMERA_STREAM_URL)

def send_stop():
    if ws: ws.send("S")

# Continuous AI Object Processing Generator Loop with Smooth Scan
def generate_video_feed():
    global control_state, HYSTERESIS
    
    room_targets = ["bottle", "cup", "chair", "laptop", "backpack", "book"]
    
    while True:
        ret, frame = video_stream.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Apply CLAHE lens filter pass to clear room glares and shadows
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame_filtered = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # FIX 2: Relaxed target parameters to catch profiles securely
        results = yolo_model(frame_filtered, verbose=False, imgsz=640, conf=0.25)
        target_found_this_frame = False
        
        delta_x = 0
        label_spotted = ""
        box_width = 0

        for result in results:
            for box in result.boxes:
                label = yolo_model.names[int(box.cls[0])].lower().strip()
                current_target = control_state["target_keyword"].lower().strip()
                
                if current_target in label or label in current_target:
                    target_found_this_frame = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_width = x2 - x1
                    delta_x = ((x1 + x2) / 2) - FRAME_CENTER
                    label_spotted = label

                    # Draw graphics feedback onto standard frame matrix
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"TRACKING: {label.upper()}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
            if target_found_this_frame: break

        # FIX 3: Hysteresis Memory Routing Block
        if target_found_this_frame:
            # Target is completely solid. Reset persistence memory counters.
            HYSTERESIS["lost_frame_counter"] = 0
            HYSTERESIS["last_known_delta_x"] = delta_x
            HYSTERESIS["last_known_label"] = label_spotted
            
            # Execute Active Navigation Control
            if control_state["nav_enabled"]:
                proximity_ratio = box_width / FRAME_WIDTH
                if proximity_ratio > 0.75:  
                    print(f"🏁 Target Reached: [{label_spotted.upper()}]. Halting chassis.")
                    send_stop()
                    time.sleep(2.5)
                    
                    try:
                        current_idx = room_targets.index(current_target)
                        next_idx = (current_idx + 1) % len(room_targets)
                    except ValueError:
                        next_idx = 0
                        
                    control_state["target_keyword"] = room_targets[next_idx]
                    print(f"🔄 Swapping target parameter to next list item: [{control_state['target_keyword'].upper()}]")
                    HYSTERESIS["last_known_label"] = ""
                else:
                    if abs(delta_x) < 25:
                        steering_bias = 0
                    else:
                        steering_bias = int(delta_x * control_state["kp"])

                    left = max(MIN_SPEED, min(control_state["base_speed"] + steering_bias, 220))
                    right = max(MIN_SPEED, min(control_state["base_speed"] - steering_bias, 220))
                    if ws: ws.send(f"M,{left},{right}")

        else:
            # Target went missing on this frame loop - evaluate memory state
            if control_state["nav_enabled"] and HYSTERESIS["lost_frame_counter"] < HYSTERESIS["max_lost_frames"] and HYSTERESIS["last_known_label"] != "":
                HYSTERESIS["lost_frame_counter"] += 1
                # Maintain the last known error trajectory vector to bridge the flicker cleanly
                smoothed_delta = HYSTERESIS["last_known_delta_x"]
                steering_bias = int(smoothed_delta * control_state["kp"])
                
                left = max(MIN_SPEED, min(control_state["base_speed"] + steering_bias, 220))
                right = max(MIN_SPEED, min(control_state["base_speed"] - steering_bias, 220))
                if ws: ws.send(f"M,{left},{right}")
            else:
                # Target is truly gone past the frame grace buffer, enter scanning sequence
                if control_state["nav_enabled"]:
                    scan = control_state["scan_speed"]
                    if ws: ws.send(f"M,{-scan},{scan}")

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# HTTP WEB SURFACE INTERFACE ENDPOINTS
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/update_sliders', methods=['POST'])
def update_sliders():
    data = request.json
    control_state["base_speed"] = int(data.get("base_speed", 90))
    control_state["kp"] = float(data.get("kp", 0.15))
    control_state["scan_speed"] = int(data.get("scan_speed", 80))
    return jsonify({"status": "success"})

@app.route('/api/toggle_ai', methods=['POST'])
def toggle_ai():
    global HYSTERESIS
    data = request.json
    control_state["nav_enabled"] = bool(data.get("nav_enabled", False))
    if not control_state["nav_enabled"]:
        send_stop()
        HYSTERESIS["lost_frame_counter"] = 0
        HYSTERESIS["last_known_label"] = ""
    return jsonify({"status": "success", "nav_enabled": control_state["nav_enabled"]})

@app.route('/api/manual_drive', methods=['POST'])
def manual_drive():
    control_state["nav_enabled"] = False 
    direction = request.json.get("direction")
    speed = control_state["base_speed"]
    
    if not ws: return jsonify({"status": "no_connection"})
    
    if direction == "F": ws.send(f"M,{speed},{speed}")
    elif direction == "B": ws.send(f"M,{-speed},{-speed}")
    elif direction == "L": ws.send(f"M,{-speed},{speed}")
    elif direction == "R": ws.send(f"M,{speed},{-speed}")
    elif direction == "S": send_stop()
    
    return jsonify({"status": "driven"})

@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    if 'audio' not in request.files:
        print("❌ Web Error: Received API packet with missing audio payload.")
        return jsonify({"status": "no_audio"})
    
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    print(f"📥 Received Audio Byte Block Size: {len(audio_bytes)} bytes")
    
    raw_path = "raw_received.blob"
    clean_wav_path = "target_voice_clean.wav"
    
    try:
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
            
        sound = AudioSegment.from_file(raw_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(clean_wav_path, format="wav")
        
        print("⚙️ Executing Whisper Neural Analysis...")
        prompt_text = "chair, bottle, cup, person, laptop, backpack, book, find, track, robot"
        prompt_ids = whisper_asr.tokenizer.get_prompt_ids(prompt_text)
        
        result = whisper_asr(
            clean_wav_path,
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "prompt_ids": torch.tensor(prompt_ids).to("cuda")
            }
        )
        text = result.get("text", "").lower().strip()
        print(f"📝 Decoded Text Output: \"{text}\"")
        
        matched_object = None
        for obj in ["chair", "bottle", "cup", "person", "laptop", "backpack", "book"]:
            if obj in text:
                control_state["target_keyword"] = obj
                matched_object = obj
                break
                
        if matched_object:
            print(f"🎯 AI Target parameter successfully switched to: [{matched_object.upper()}]")
        else:
            print("⚠️ Transcription did not match any supported targets.")
            
    except Exception as e:
        print(f"❌ Audio Processing Pipeline Failure: {e}")
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(clean_wav_path): os.remove(clean_wav_path)
            
    return jsonify({"status": "processed", "target": control_state["target_keyword"].upper()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)