import cv2
from ultralytics import YOLO

print("🧠 Loading YOLOv8 Lightweight Vision Model...")
# This automatically downloads a small, high-speed 4k-object detector (~6MB)
model = YOLO("yolov8n.pt") 
print("Model loaded successfully!")

# 🛑 REPLACE THIS URL with the exact IP address shown on your phone's app!
# Note: Keep the '/video' or '/video.jpg' suffix depending on what the app specifies
CAMERA_URL = "http://10.230.238.88:8080/video"

print(f"🔌 Connecting to smartphone camera stream at: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ Error: Could not open the video stream. Check your IP address and Wi-Fi connection.")
    exit()

print("\n--- VISION SYSTEM ACTIVE ---")
print("Press 'q' on your keyboard while looking at the camera window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Run the image frame through the YOLO AI (stream=True optimizes processing)
    results = model(frame, stream=True)

    # Look through everything the AI detects in the room
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the name of the object (e.g., "chair", "bottle", "person")
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            
            # Get the exact center coordinate of the object on the screen
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = int((x1 + x2) / 2)
            
            # Print targeted object locations to the terminal
            if object_name in ["chair", "bottle", "cup", "person"]:
                print(f"🎯 Spotted: {object_name.upper()} | Screen Center X: {x_center}")

        # Show the live video feed with bounding boxes drawn around objects
        annotated_frame = r.plot()
        cv2.imshow("Robot Eyes - YOLOv8 Live Feed", annotated_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[SHUTDOWN] Vision tracking stopped.")