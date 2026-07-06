import numpy as np
import sounddevice as sd
import time
from transformers import pipeline

# --- NEW: Import our Virtual Arduino instead of the real serial port ---
from virtual_arduino import mock_arduino as arduino
print("Connected to Virtual Arduino successfully!")

# Load Hugging Face Speech Recognition Model (Whisper-Base)
print("Loading Hugging Face Whisper-Base Model...")
asr_pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-base", 
    device="cpu"
)
print("Model loaded successfully!")

SAMPLING_RATE = 16000
DURATION = 2.5  
VOLUME_THRESHOLD = 0.022  


def listen_and_command():
    print("\nListening for command... (Speak now)")
    audio_data = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype="float32")
    sd.wait()  

    audio_inputs = np.squeeze(audio_data)
    rms_volume = np.sqrt(np.mean(audio_inputs**2))

    if rms_volume < VOLUME_THRESHOLD:
        print("Skipping AI processing: It's too quiet.")
        return  

    prediction = asr_pipe(audio_inputs)
    text = prediction["text"].lower().strip()
    print(f"Recognized Text: '{text}'")

    command = None
    if any(word in text for word in ["forward", "forwad", "go", "front"]):
        command = "F"
    elif any(word in text for word in ["back", "backward", "reverse", "revers"]):
        command = "B"
    elif any(word in text for word in ["left", "let", "leftside"]):
        command = "L"
    elif any(word in text for word in ["right", "write", "rait"]):
        command = "R"
    elif any(word in text for word in ["stop", "halt", "break"]):
        command = "S"
    elif "exit" in text:
        command = "EXIT"

    if command == "EXIT":
        print("\n[SHUTDOWN] Exit keyword detected.")
        raise KeyboardInterrupt
    elif command:
        # --- NEW: Send byte data to the simulation ---
        arduino.write(command.encode())
    else:
        print("Loud noise detected, but no matching direction keyword found.")


if __name__ == "__main__":
    while True:
        try:
            listen_and_command()
        except KeyboardInterrupt:
            print("\nProgram stopped cleanly.")
            break