import serial
import time
import sys
from faster_whisper import WhisperModel

# ==========================================
# CONFIGURATION
# ==========================================
# IMPORTANT: Double-check your Outgoing COM port number in Bluetooth settings!
SERIAL_PORT = 'COM8' 
BAUD_RATE = 115200

# ==========================================
# INITIALIZE WIRELESS BLUETOOTH CONNECTION
# ==========================================
print("🤖 Initializing AI Voice Robot Control System...")
print(f"📡 Attempting connection to robot on {SERIAL_PORT}...")

ser = None
for attempt in range(1, 4):
    try:
        time.sleep(1) # Give the system antenna a moment to settle
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=2,        # Gives the Bluetooth stack more time to respond
            rtscts=False,     # Disables hardware flow control to prevent terminal lockups
            dsrdtr=False
        )
        print("✅ Connected successfully over Bluetooth airwaves!")
        break
    except serial.SerialException as e:
        print(f"⚠️ Connection attempt {attempt} failed: {e}")
        if attempt < 3:
            print("Retrying connection in 2 seconds...")
            time.sleep(2)
        else:
            print("\n❌ Error: Could not open the Bluetooth port after multiple attempts.")
            print("💡 Fixes: Re-pair the ESP32 in Windows settings, close other background terminals, or check if the Outgoing COM port number changed.")
            sys.exit(1)

# ==========================================
# INITIALIZE AI SPEECH ENGINE
# ==========================================
print("\n🧠 Loading Faster-Whisper AI Model (tiny.en)...")
# Using the tiny model for real-time speed on local machines
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("🎙️ AI Engine Ready! Start speaking commands (e.g., 'Forward', 'Stop')...")

# ==========================================
# MAIN CORE VOICE PROCESSING LOOP
# ==========================================
try:
    while True:
        # Note: In a full deployment, this would actively capture audio from a live stream.
        # This wrapper safely processes identified commands.
        print("\n[Listening for voice command... Input text simulated below]")
        
        # Taking inputs to simulate the speech detection path for testing the wireless bridge
        command_text = input("Enter simulated voice command: ").strip().lower()
        
        if "forward" in command_text:
            print("🚀 Command Detected: FORWARD -> Sending 'F' wirelessly...")
            try:
                ser.write(b'F')
                ser.flush()  # Forces the data out of the laptop buffer immediately
            except Exception as e:
                print(f"❌ Failed to transmit data packet: {e}")
                
        elif "stop" in command_text:
            print("🛑 Command Detected: STOP -> Sending 'S' wirelessly...")
            try:
                ser.write(b'S')
                ser.flush()
            except Exception as e:
                print(f"❌ Failed to transmit data packet: {e}")
                
        elif "exit" in command_text or "quit" in command_text:
            print("Shutting down voice system...")
            break
            
        else:
            print("❓ Command not recognized. Try saying 'Forward' or 'Stop'.")

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    if ser and ser.is_open:
        ser.close()
        print("🔌 Bluetooth serial port closed cleanly.")