import hid
import time

# SpaceMouse VID and PID details
vendor_id = 0x256f
product_id = 0xc638

device = hid.device()
device.open(vendor_id, product_id)
print("Connected to SpaceMouse Pro!")

# keep running even when no data is sent
device.set_nonblocking(True)

def interpret_motion(data):
    """
    Given 13 byte data from the device (important), do 3 things:
    1. Read in the x, y, z, and corresponding rotation values
    2. interpret how far user moved mouse from its original position
    3. Print to the console/terminal

    Caveat: Spacemouse must be plugged in via USB - data is "hidden" via
    bluetooth AND the reciever unfortunately
    """
# navigate into serve then src after opening live server in commands
    if len(data) < 7:
        return

    # Translation (movement) data
    x = int.from_bytes(data[1:3], byteorder='little', signed=True)
    y = int.from_bytes(data[3:5], byteorder='little', signed=True)
    z = int.from_bytes(data[5:7], byteorder='little', signed=True)

    # Rotation data (pitch, yaw, roll)
    rot_x = int.from_bytes(data[7:9], byteorder='little', signed=True)
    rot_y = int.from_bytes(data[9:11], byteorder='little', signed=True)
    rot_z = int.from_bytes(data[11:13], byteorder='little', signed=True)

    threshold = 100  # Ignore very small noise
    rotation_threshold = 200  # Rotation threshold

    directions = []

    # Translation (movement)
    if abs(x) > threshold:
        direction = "right" if x > 0 else "left"
        directions.append(f"Moved {direction}")
    if abs(y) > threshold:
        direction = "backward" if y > 0 else "forward"
        directions.append(f"Moved {direction}")
    if abs(z) > threshold:
        direction = "down" if z > 0 else "up"
        directions.append(f"Moved {direction}")

    # Rotation (pitch, yaw, roll)
    if abs(rot_x) > rotation_threshold:
        direction = "clockwise" if rot_x > 0 else "counterclockwise"
        directions.append(f"Rotated pitch {direction}")
    if abs(rot_y) > rotation_threshold:
        direction = "clockwise" if rot_y > 0 else "counterclockwise"
        directions.append(f"Rotated yaw {direction}")
    if abs(rot_z) > rotation_threshold:
        direction = "clockwise" if rot_z > 0 else "counterclockwise"
        directions.append(f"Rotated roll {direction}")

    # If multiple directions at the same time
    if directions:
        print(", ".join(directions))

# Running at all times
while True:
    report = device.read(13) # 13 byte data
    if report:
        interpret_motion(report) # only interpret WHEN data is sent
    time.sleep(0.01)
