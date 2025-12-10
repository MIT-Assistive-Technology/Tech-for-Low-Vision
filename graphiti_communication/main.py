import time

import serial
from graphiticommands import enable_touch_mode

# Configure and open the serial port
port = serial.Serial(
    port="COM3",  # Compatible with windows, Change to your serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows)
    baudrate=115200,
    timeout=2,
)

# Ensure the port is open
if port.is_open:
    print(f"Connected to {port.name}")

# Send a message
message = enable_touch_mode().encode()
port.write(message)

# Read a response (if any)
time.sleep(2)
response = port.readline()
print(f"Received: {response}")

port.close()
