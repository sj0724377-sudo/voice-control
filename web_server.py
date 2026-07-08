import os
import time
import cv2
import torch
import websocket
import threading
import numpy as np
import socket  # Auto-detect your Wi-Fi IP address
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from transformers import pipeline
from pydub import AudioSegment

app = Flask(__name__)

ESP32_IP = "10.206.57.189"          
CAMERA_STREAM_URL = "http://10.206.57.139:8080/video"  

FRAME_WIDTH = 640
FRAME_CENTER = 320                  
MIN_SPEED = 50                      
PROXIMITY_BRAKE_RATIO = 0.35  # 🚨 NEW: Lowered from 0.75! (0.45 = stops further away, 0.80 = gets very close)

# Global Dynamic Tracking States shared across network handlers
control_state = {
    "nav_enabled": False,
    "target_keyword": None,
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

# 🚨 UPDATED BLOCK: Custom trained colored boxes!
ROOM_TARGETS = ['red', 'pink', 'green', 'orange']
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

# Load your newly compiled custom YOLO brain!
yolo_model = YOLO("best.pt").to("cuda")

print("⚡ Loading OpenAI Whisper-Small model in FP16 precision...")
# 🚨 PERFORMANCE UPGRADE: Using whisper-small with FP16 (Half-Precision) for maximum RTX 3050 speed
whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device=0,
    torch_dtype=torch.float16
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

# Continuous AI Object Processing Generator Loop
def generate_video_feed():
    global control_state, HYSTERESIS
    
    while True:
        ret, frame = video_stream.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Guard check! If no target is specified yet, do not run YOLO or track anything
        if control_state["target_keyword"] is None:
            cv2.putText(frame, "AI TARGET: NOT SPECIFIED (STANDBY)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
            
            # Ensure motors stop immediately if navigation was accidentally toggled on
            if control_state["nav_enabled"]:
                send_stop()
                control_state["nav_enabled"] = False
                
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            continue

        # Smoothed lens filter pass to clear shadows
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame_filtered = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Baseline 0.40 confidence gate for stable tracking
        results = yolo_model(frame_filtered, verbose=False, imgsz=640, conf=0.40)
        target_found_this_frame = False
        
        delta_x = 0
        label_spotted = ""
        box_width = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id].lower().strip()
                current_target = control_state["target_keyword"].lower().strip()
                
                # Broadened check: This maps "green" (spoken) to "green_box" (YOLO output)
                if current_target in label or label in current_target:
                    target_found_this_frame = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_width = x2 - x1
                    delta_x = ((x1 + x2) / 2) - FRAME_CENTER
                    label_spotted = label

                    # Draw standard green feedback graphics onto your stream matrix
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"TRACKING: {label.upper()}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
            if target_found_this_frame: break

        # Hysteresis Memory Routing Block
        if target_found_this_frame:
            HYSTERESIS["lost_frame_counter"] = 0
            HYSTERESIS["last_known_delta_x"] = delta_x
            HYSTERESIS["last_known_label"] = label_spotted
            
            # Execute Active Navigation Control
            if control_state["nav_enabled"]:
                proximity_ratio = box_width / FRAME_WIDTH
                
                # 🚨 FIX: Replaced static 0.75 with the PROXIMITY_BRAKE_RATIO variable here!
                if proximity_ratio > PROXIMITY_BRAKE_RATIO:  
                    send_stop()  
                    control_state["nav_enabled"] = False
                    HYSTERESIS["last_known_label"] = ""
                    HYSTERESIS["lost_frame_counter"] = 0
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

        if not control_state["nav_enabled"] and HYSTERESIS["lost_frame_counter"] == 0 and label_spotted != "":
            cv2.putText(frame, "TARGET REACHED - WAITING FOR COMMAND", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

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

@app.route('/api/get_status', methods=['GET'])
def get_status():
    target_str = control_state["target_keyword"].upper() if control_state["target_keyword"] else "NOT SPECIFIED"
    return jsonify({
        "nav_enabled": control_state["nav_enabled"],
        "target": target_str,
        "base_speed": control_state["base_speed"],
        "kp": control_state["kp"],
        "scan_speed": control_state["scan_speed"]
    })

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
    desired_state = bool(data.get("nav_enabled", False))
    
    if desired_state and control_state["target_keyword"] is None:
        send_stop()
        control_state["nav_enabled"] = False
        return jsonify({"status": "no_target", "message": "Please specify a target first!", "nav_enabled": False})

    control_state["nav_enabled"] = desired_state
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
    
    raw_path = "raw_received.blob"
    clean_wav_path = "target_voice_clean.wav"
    
    try:
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
            
        sound = AudioSegment.from_file(raw_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(clean_wav_path, format="wav")
        
        print("⚙️ Executing Whisper Neural Analysis...")
        # 🚨 UPDATED PROMPT: Giving Whisper the hint for your exact custom colors
        prompt_text = "red box, pink box, green box, orange box, red, pink, green, orange, find, track, robot"
        prompt_ids = whisper_asr.tokenizer.get_prompt_ids(prompt_text)
        
        # Kept "language": "en" for whisper-small since it is a multilingual model
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
        for obj in ROOM_TARGETS:
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
            
    current_target_str = control_state["target_keyword"].upper() if control_state["target_keyword"] else "NOT SPECIFIED"
    return jsonify({"status": "processed", "target": current_target_str})

# =====================================================================
# 🌐 CUSTOM FRIENDLY NETWORK LAUNCHER
# =====================================================================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    wifi_ip = get_local_ip()
    port = 5000

    print("\n" + "="*65)
    print(" 🚀 ESP32 AI ROBOTICS - GROUND CONTROL STATION ONLINE!")
    print("="*65)
    print(" HOW TO CONNECT TO YOUR DASHBOARD:")
    print("")
    print(" 💻 1. IF YOU ARE ON THIS LAPTOP:")
    print(f"       👉 Open your browser to: http://localhost:{port}")
    print("       (Uses direct internal loopback - fastest connection)")
    print("")
    print(" 📱 2. IF YOU ARE USING A PHONE, TABLET, OR ANOTHER PC:")
    print(f"       👉 Open their browser to: http://{wifi_ip}:{port}")
    print("       (Must be connected to the exact same Wi-Fi network!)")
    print("="*65 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)