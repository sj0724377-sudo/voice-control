import cv2
import serial
import time
from ultralytics import YOLO

print("🔌 Connecting to Arduino on COM8...")
try:
    # Open the real serial link to your new Arduino
    arduino = serial.Serial('COM8', 9600, timeout=0.1)
    time.sleep(2)  # Give Arduino time to reset after connecting
    print("✅ Arduino connected successfully!")
except Exception as e:
    print(f"❌ Error connecting to Arduino: {e}")
    exit()

print("🧠 Loading YOLOv8 Vision Model...")
model = YOLO("yolov8s.pt") 

# 🛑 Double-check this matches your phone's stream IP!
CAMERA_URL = "http://10.230.238.88:8080/video"
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ Error: Could not open the phone camera stream.")
    arduino.close()
    exit()

# Define the target object you want the car to follow
TARGET_OBJECT = "chair" 

print(f"\n🚀 SYSTEM ACTIVE: Tracking and moving toward any '{TARGET_OBJECT}'...")
print("Press 'q' in the video window to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get image dimensions (typically 1280x720 or 640x480)
    height, width, _ = frame.shape
    screen_center = width // 2
    
    # Define dead-zones around the center (adjust padding to tune steering sensitivity)
    padding = int(width * 0.15)  # 15% width buffer on left/right
    left_bound = screen_center - padding
    right_bound = screen_center + padding

    results = model(frame, stream=True, conf=0.25)
    target_spotted = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            
            if object_name == TARGET_OBJECT:
                target_spotted = True
                
                # Calculate coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x_center = int((x1 + x2) / 2)
                box_height = y2 - y1
                
                # 1. Path Planning & Steering Logic
                if x_center < left_bound:
                    print(f"🔄 Target Left ({x_center}) -> Sending 'L'")
                    arduino.write(b'L')
                elif x_center > right_bound:
                    print(f"🔄 Target Right ({x_center}) -> Sending 'R'")
                    arduino.write(b'R')
                else:
                    # 2. Distance/Stopping Logic: If it takes up > 60% of the screen height, stop.
                    if box_height > (height * 0.60):
                        print("🛑 Arrived at destination! -> Sending 'S'")
                        arduino.write(b'S')
                    else:
                        print(f"🏎️ Target Centered ({x_center}) -> Sending 'F'")
                        arduino.write(b'F')
                break # Target the first instance found
        
        # Draw bounding boxes and zone lines on visual feed
        annotated_frame = r.plot()
        cv2.line(annotated_frame, (left_bound, 0), (left_bound, height), (0, 0, 255), 2)
        cv2.line(annotated_frame, (right_bound, 0), (right_bound, height), (0, 0, 255), 2)
        cv2.imshow("Robot Tracking Eyes", annotated_frame)

    # If the target object isn't anywhere in the frame, tell the car to stop searching blindly
    if not target_spotted:
        print("🔍 Searching for target... -> Sending 'S'")
        arduino.write(b'S')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up connections on exit
print("\nShutting down systems...")
arduino.write(b'S') # Ensure motors stop
arduino.close()
cap.release()
cv2.destroyAllWindows()