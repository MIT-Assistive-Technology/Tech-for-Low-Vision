import serial
import time
from graphiticommands import *

# Configure and open the serial port
ser = serial.Serial(
    port='COM3',  # Compatible with windows, Change to your serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows)
    baudrate=115200,
    timeout=2 
)

# Ensure the port is open
if ser.is_open:
    print(f"Connected to {ser.name}")

# Send a message
message = enable_touch_mode()
ser.write(message)

# Read a response (if any)
time.sleep(2)
response = ser.readline() 
print(f"Received: {response}")

ser.close()