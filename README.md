##  Code & Environment Requirements

To run this backend control engine, your local environment must meet the following software, library, and hardware acceleration prerequisites.

### 1. Core Runtime Environment
* **Python Engine:** Python 3.8 to Python 3.11 (Python 3.10 recommended for stable CUDA mapping).
* **Hardware Acceleration API:** NVIDIA CUDA Toolkit (v11.8 or v12.1) along with matching cuDNN libraries to enable real-time RTX 3050 GPU processing.

### 2. Core Dependencies & Frameworks
The project relies on the following primary Python library ecosystems:

* **Web & Networking Frameworks:**
  * `Flask` (Core web backend application handling control routes)
  * `Flask-SocketIO` / `python-socketio` (Manages ultra-low-latency WebSocket channels to the ESP32)

* **Computer Vision & Deep Learning Execution:**
  * `ultralytics` (Engine powering the real-time YOLOv8 object detection)
  * `opencv-python` (Handles incoming MJPEG stream capturing, matrix resizing, and CLAHE optical filtering)
  * `torch` & `torchvision` (PyTorch compiled with CUDA support to offload vision arrays to VRAM)

* **Automatic Speech Recognition (ASR):**
  * `openai-whisper` (Offline transcription pipeline processing voice command blobs)
  * `numpy` (Manages audio sampling matrix conversions)

### 3. Required Local Artifacts (Not included in repo)
Because heavy files are excluded via `.gitignore`, you must manually place the following files into your project root directory before spinning up the server:
* **Custom Weights Matrix:** `best.pt` (Your custom-trained YOLOv8 model weights file).
# 🏎️ ESP32 AI Robotics Car Controller

This repository contains the full Edge-AI vision and voice processing control suite for an autonomous tracking robot car. The system leverages local hardware acceleration to run deep learning models seamlessly without reliance on external cloud APIs.

## 🚀 Core Entry Point
* **`web_server.py`**: **[PRIMARY SYSTEM ENGINE]** Run this script to initialize the local Flask interface, establish ultra-low-latency WebSocket channels with the ESP32 chassis gateway, process incoming camera matrices on your dedicated GPU, and process voice inputs via local ASR.

---

## 🛠️ Quick Start
To spin up the server environment on your local network architecture, execute:
```bash
python web_server.py

```

---

## 📐 System Architecture & Data Flow

The robot functions as a closed-loop cyber-physical system split into three distinct computing layers:

```text
[ ESP32 Cam / IP Stream ] --( Real-time MJPEG HTTP Stream )--> [ Local Python Server ]
                                                                      |
                                                               ( RTX 3050 CUDA )
                                                                      |
                                                          +-----------+-----------+
                                                          |                       |
                                                   [ YOLOv8 Vision ]      [ Whisper ASR ]
                                                    (Object Bounding)     (Voice Controls)
                                                          |                       |
                                                    (Pixel Delta)         (Target Keyword)
                                                          |                       |
                                                          +-----------+-----------+
                                                                      |
                                                            [ P-Controller Vector ]
                                                                      |
[ ESP32 Car Chassis ] <--( Low-Latency WebSockets Control )----------+

```

### 1. The Perceptual Input Layer (Vision & Voice)

* **Live Camera Interface:** An external IP camera stream continuously pipes 640x480 MJPEG video matrices over HTTP to the Flask backend thread.
* **Optical Filter Pass (CLAHE):** Incoming frames pass through a Contrast Limited Adaptive Histogram Equalization filter in OpenCV. This stabilizes sudden glares and balances room shadows, ensuring objects like fans or beds match the structural appearance of the training dataset.
* **Edge-AI Object Detection:** The frame is sent directly to an optimized **YOLOv8** model running locally on the **NVIDIA RTX 3050 GPU**. Detections filter down through a `0.40` confidence gate across a 6-class alphabetical vocabulary matrix (`['bed', 'bottle', 'chair', 'fan', 'sofa', 'table']`).
* **Local Speech Recognition:** Audio recorded from the browser control surface (`.blob`) is packaged and sent to an offline **OpenAI Whisper Pipeline**. Whisper runs directly on GPU VRAM, transcribing voice commands using custom keyword token arrays to dynamically swap the robot's tracking keyword on the fly.

### 2. The Kinematic Control Layer (Pixel-Based Tracking)

The steering mechanism utilizes a pixel-space **Proportional (P) Controller** to convert coordinates into motor adjustments:

* **Error Isolation:**
The system establishes the camera center at pixel column **320** (`FRAME_CENTER = 640 / 2`). The controller constantly tracks the horizontal center point of the object's bounding box and computes the exact pixel deviation (`delta_x`):
```text
delta_x = ((x1 + x2) / 2) - 320

```


* **Proportional Vectoring:**
This error distance is multiplied by a proportional steering gain (`kp = 0.15`) to scale the steering response:
```text
steering_bias = int(delta_x * kp)

```


* **Differential Speed Mapping:**
The bias values adjust the independent left and right wheel speeds relative to a baseline forward velocity:
```text
Left Motor Speed = base_speed + steering_bias
Right Motor Speed = base_speed - steering_bias

```



### 3. Safety, Memory, & Mechanical Execution

* **Hysteresis Smoothing Buffers:** To prevent jerky maneuvers if the vision model briefly drops a tracking frame due to rapid turns or motion blur, a frame grace buffer stores the last known vector coordinates. The vehicle seamlessly rides through frame drops without breaking tracking lock or dropping into its default rotation search mode.
* **Proximity Braking:** The controller constantly computes the percentage of the screen filled by the object's boundary width. The exact millisecond the proximity ratio breaks past **`0.75`** (meaning the target occupies more than 75% of the frame width), an automatic emergency brake vector (`S`) triggers to safely park the chassis right in front of the object.
* **WebSocket Network Uplink:** Motor speed integers are structured into high-speed command strings (`M,left,right`) and shot across a low-overhead **WebSocket channel (`ws://`)** directly to the listening ESP32 gateway router on the car, achieving near-instantaneous mechanical responses.

```

```
