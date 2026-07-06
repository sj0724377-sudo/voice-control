import websocket
import sys
import time

# ==================== CONFIGURATION ====================
ESP32_IP_RAW = "10.230.238.189"  # <--- Put your active Serial Monitor IP here
WS_URL = f"ws://10.230.238.189:81" 
# =======================================================

print("Connecting to ESP32 WebSocket server for motion testing...")
try:
    ws = websocket.WebSocket()
    ws.connect(WS_URL, timeout=3.0)
    print("🔌 Connected successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}. Confirm the ESP32 is powered on and matches your IP.")
    sys.exit()

print("\n==============================================")
print("             MOTOR DIAGNOSTIC MENU            ")
print("==============================================")
print("Press the following keys in your terminal and hit Enter:")
print("  f -> Test FORWARD Motion (Both motors run at 95)")
print("  l -> Test LEFT Pivot Turn (Right motor runs at 80, Left locks)")
print("  r -> Test RIGHT Pivot Turn (Left motor runs at 80, Right locks)")
print("  s -> Send STOP Command (All motors brake)")
print("  q -> Quit testing utility")
print("==============================================\n")

try:
    while True:
        # Request key command via console input
        user_input = input("Enter command (f/l/r/s/q): ").strip().lower()
        
        if user_input == 'f':
            print("🚀 Sending: FORWARD")
            ws.send('F')
        elif user_input == 'l':
            print("◀ Sending: LEFT PIVOT")
            ws.send('L')
        elif user_input == 'r':
            print("▶ Sending: RIGHT PIVOT")
            ws.send('R')
        elif user_input == 's':
            print("■ Sending: STOP")
            ws.send('S')
        elif user_input == 'q':
            print("Closing utility...")
            ws.send('S')
            break
        else:
            print("⚠️ Invalid key. Use f, l, r, s, or q.")
            
except KeyboardInterrupt:
    print("\nAborting...")
finally:
    try:
        ws.send('S')
        ws.close()
    except Exception:
        pass
    print("Test connection closed cleanly.")