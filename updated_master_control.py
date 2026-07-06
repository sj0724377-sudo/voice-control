import cv2
import torch
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import requests
import time
from ultralytics import YOLO
from transformers import pipeline

# ==================== CONFIGURATION ====================
ESP32_IP = "http://10.230.238.189"  # Replace with your ESP32 local IP
CAMERA_URL = "http://10.230.238.88:8080/video"  # Replace with your IP Webcam stream URL
AUDIO_FILENAME = "command.wav"
SAMPLING_RATE = 16000
RECORD_DURATION = 5  # Seconds to listen to your voice command
# =======================================================

print("Initializing AI Models... Please wait...")
yolo_model = YOLO("yolov8n.pt")  # Lightweight Nano model
# Force language='en' and give it a hint task to prevent hallucinations in silent rooms
whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny",
    generate_kwargs={"language": "en", "task": "transcribe"}
)

def send_robot_command(action):
    """Sends a non-blocking network request to control the physical chassis."""
    try:
        url = f"{ESP32_IP}/{action}"
        requests.get(url, timeout=0.5)
    except requests.exceptions.RequestException:
        pass

def record_voice_command():
    """Records audio from laptop microphone."""
    print("\n🎤 Listening for command... Speak now!")
    audio_data = sd.rec(int(RECORD_DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='int16')
    sd.wait()
    print("🎙️ Recording stopped. Processing speech...")
    wavfile.write(AUDIO_FILENAME, SAMPLING_RATE, audio_data)

def parse_target_object():
    """Converts the recorded audio data directly using Whisper, bypassing FFmpeg."""
    # 1. Record the audio array
    print("\n🎤 Listening for command... Speak now!")
    audio_data = sd.rec(int(RECORD_DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
    sd.wait()
    print("🎙️ Recording stopped. Processing speech...")
    
    # 2. Flatten the array into a 1D vector required by Hugging Face
    audio_input = audio_data.flatten()
    
    # 3. Pass the raw numpy array and explicit sampling rate to the pipeline
    result = whisper_asr({"raw": audio_input, "sampling_rate": SAMPLING_RATE})
    text_command = result["text"].lower()
    print(f"📄 Decoded Command text: '{text_command}'")
    
    # Target keyword parsing extraction
    if "chair" in text_command:
        return "chair"
    elif "bottle" in text_command:
        return "bottle"
    elif "person" in text_command:
        return "person"
    else:
        print("⚠️ Target item not recognized in text string.")
        return None
    
def navigate_to_target(target_label):
    """Closes control loops tracking objects in frame coordinates."""
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Cannot connect to mobile camera video stream!")
        return

    print(f"🔍 Initiating environmental scanning sequence for: [{target_label}]")
    send_robot_command("left")  # Starts spinning at a controlled 50% speed
    
    target_found = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_height, frame_width, _ = frame.shape
        frame_center_x = frame_width // 2
        
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes
        
        best_box = None
        for box in detections:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            if label == target_label:
                best_box = box
                break
                
        if best_box is not None:
            if not target_found:
                print("🎯 Target spotted! Locking target trajectory coordinates...")
                send_robot_command("stop")
                time.sleep(0.2)
                target_found = True
                
            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            box_center_x = (x1 + x2) // 2
            
            box_width = x2 - x1
            proximity_ratio = box_width / frame_width
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking: {target_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            error_margin = 50 
            if proximity_ratio > 0.40:  # Triggers stop when target occupies >40% of frame width
                print("🏁 Target reached successfully! HALTING.")
                send_robot_command("stop")
                break
            elif box_center_x < (frame_center_x - error_margin):
                send_robot_command("left")
            elif box_center_x > (frame_center_x + error_margin):
                send_robot_command("right")
            else:
                send_robot_command("forward")
        else:
            if target_found:
                print("❓ Lost sight of target tracking marker. Scanning again...")
                send_robot_command("left")
                target_found = False
                
        cv2.imshow("Robot AI Vision Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    send_robot_command("stop")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target = parse_target_object()
    if target:
        navigate_to_target(target)
    else:
        print("❌ Pipeline execution abandoned: No identifiable target target tokens.")