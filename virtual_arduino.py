import time
import os

class VirtualArduino:
    def __init__(self):
        self.current_state = "STOPPED"
        print("🤖 [VIRTUAL ARDUINO] Booting up ATmega328P simulation...")
        time.sleep(1)
        print("🔌 [VIRTUAL ARDUINO] Virtual Serial Port Open on Mock Channel.")
        print("Waiting for commands from speech_to_text.py...\n")

    def write(self, data):
        # Decode the incoming byte command from your speech script
        command = data.decode().strip()
        
        if command == "F":
            self.current_state = "🚗 MOVING FORWARD (^)"
        elif command == "B":
            self.current_state = "🚙 REVERSING (v)"
        elif command == "L":
            self.current_state = "👈 TURNING LEFT (<)"
        elif command == "R":
            self.current_state = "👉 TURNING RIGHT (>)"
        elif command == "S":
            self.current_state = "🛑 STOPPED"
            
        # Draw a little dashboard in your terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=========================================")
        print("       VIRTUAL ROBOTIC CAR DASHBOARD     ")
        print("=========================================")
        print(f" Received Byte Signal: '{command}'")
        print(f" Current Car State   : {self.current_state}")
        print("=========================================")

# Create a global instance we can import
mock_arduino = VirtualArduino()