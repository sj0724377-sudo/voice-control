import numpy as np
import sounddevice as sd

SAMPLING_RATE = 16000
DURATION = 2.5  # Seconds

print("\n=== MICROPHONE ACCURACY DIAGNOSTIC ===")
print("System is listening... Speak a command clearly (e.g., 'FORWARD') now!")


audio_data = sd.rec(
    int(DURATION * SAMPLING_RATE),
    samplerate=SAMPLING_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()  
print("Audio buffer captured successfully.")


audio_inputs = np.squeeze(audio_data)


volume_level = np.sqrt(np.mean(audio_inputs**2))

print("-" * 38)
print(f"Calculated Sound Energy Level: {volume_level:.4f}")
print("-" * 38)

if volume_level > 0.010:
    print("STATUS: SUCCESS! Your microphone is capturing clean audio levels.")
else:
    print("STATUS: WARNING! Energy too low. Check if your mic is muted or quiet.")