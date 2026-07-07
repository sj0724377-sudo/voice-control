# 🏎️ ESP32 AI Robotics Car Controller

This repository contains the full Edge-AI vision and voice processing control suite for an autonomous tracking robot car. The system leverages local hardware acceleration to run deep learning models seamlessly without reliance on external cloud APIs.

## 🚀 Core Entry Point
* **`web_server.py`**: **[PRIMARY SYSTEM ENGINE]** Run this script to initialize the local Flask interface, establish ultra-low-latency WebSocket channels with the ESP32 chassis gateway, process incoming camera matrices on your dedicated GPU, and process voice inputs via local ASR.

---

## 📦 Code & Environment Prerequisites

To run this backend control engine, your local environment must meet the following software, library, and hardware acceleration prerequisites:

### 1. Core Runtime Environment
* **Python Engine:** Python 3.8 to Python 3.11 (Python 3.10 is recommended for stable CUDA mapping).
* **Hardware Acceleration API:** NVIDIA CUDA Toolkit (v11.8 or v12.x) along with matching cuDNN libraries to enable real-time RTX 3050 GPU processing.

### 2. Core Dependencies & Frameworks
The project relies on the following primary Python library ecosystems (all mapped inside `requirements.txt`):
* **Web & Networking Frameworks:** `Flask==3.1.3`, `websocket-client==1.9.0`, `Werkzeug==3.1.8`
* **Computer Vision & Deep Learning Execution:** `ultralytics==8.4.70`, `opencv-python==4.13.0.92`, `torch==2.12.1+cu126`, `torchvision==0.27.1+cu126`
* **Automatic Speech Recognition (ASR):** `openai-whisper==20250625`, `faster-whisper==1.2.1`, `pydub==0.25.1`, `numpy==2.4.6`

### 3. Required Local Artifacts (Not included in repo)
Because heavy model files are excluded via `.gitignore`, you must manually place the following files into your project root directory before spinning up the server:
* **Custom Weights Matrix:** `best.pt` (Your custom-trained YOLOv8 model weights file).

---

## 🛠️ Setup Procedure & How to Run

### 1. Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use an isolated Python environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Required Dependencies

Install the entire project library suite using the provided requirements file:

```bash
pip install -r requirements.txt

```

### 4. Run the Ground Control Server

Ensure your ESP32 robot chassis and IP camera are powered on and connected to your local Wi-Fi network, then execute:

```bash
python web_server.py

```

Once the server initializes and detects your local Wi-Fi IP, open your web browser and navigate to `http://localhost:5000` (if on the host PC) or the LAN IP displayed in your terminal (e.g., `http://10.206.57.152:5000`) to access the Ground Control Station dashboard!

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
