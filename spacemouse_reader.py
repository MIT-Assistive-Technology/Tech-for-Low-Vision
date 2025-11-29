import hid
import time
import requests

# SpaceMouse VID and PID details
vendor_id = 0x256f
product_id = 0xc638

device = hid.device()
device.open(vendor_id, product_id)
print("Connected to SpaceMouse Pro!")
SERVER_URL = "http://localhost:3000/api/spacemouse"

# keep running even when no data is sent
device.set_nonblocking(True)

def interpret_motion(data):
    """"
    Reads motion data, interprets it, and SENDS it to the server.
    """
    if len(data) < 13: # SpaceMouse Pro/Wireless sends a 13-byte report for motion
        return

    # Translation (movement) data
    x = int.from_bytes(data[1:3], byteorder='little', signed=True)
    y = int.from_bytes(data[3:5], byteorder='little', signed=True)
    z = int.from_bytes(data[5:7], byteorder='little', signed=True)

    # Rotation data (pitch, yaw, roll)
    rot_x = int.from_bytes(data[7:9], byteorder='little', signed=True)
    rot_y = int.from_bytes(data[9:11], byteorder='little', signed=True)
    rot_z = int.from_bytes(data[11:13], byteorder='little', signed=True)

    # NEW: Prepare the data payload to send to the server
    payload = {
        "x": x,
        "y": y,
        "z": z,
        "rot_x": rot_x,
        "rot_y": rot_y,
        "rot_z": rot_z
    }

    # Optional: Keep console print for debugging
    # print(f"Raw Data: x={x}, y={y}, z={z}, rot_x={rot_x}, rot_y={rot_y}, rot_z={rot_z}")

    # NEW: Send the data to the Node.js server
    try:
        # Use a timeout to prevent the script from hanging if the server is down
        response = requests.post(SERVER_URL, json=payload, timeout=0.5)
        # Optional: Print response status for confirmation
        if not response.ok:
            print(f"Failed to send data: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to server: {e}") # Uncomment for server connection errors
        pass # Ignore errors to keep the SpaceMouse script running smoothly


# Running at all times
while True:
    report = device.read(13) # 13 byte data
    if report:
        # Only interpret and send data when a report is available
        interpret_motion(report)
    # Use a small sleep to prevent excessive CPU usage
    time.sleep(0.01)

# use this C:\Users\megp_\AppData\Local\Programs\Python\Python311\python.exe spacemouse_reader.py to run mouse controls on this terminals
# open up a new terminal powershell to run node server.js
