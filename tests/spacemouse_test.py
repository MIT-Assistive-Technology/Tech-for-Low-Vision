# import pyspacemouse
# import time
# import os
# import ctypes

# # Get the folder of this script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Prepend it to PATH so DLLs can be found
# os.environ['PATH'] = script_dir + os.pathsep + os.environ['PATH']

# # Optional: explicitly load hidapi.dll to make sure it's found
# ctypes.CDLL(os.path.join(script_dir, "hidapi.dll"))



# def main():
#     sm = pyspacemouse.open()

#     print("Spacemouse opened successfully."
#     )

#     try:
#         while True:
#             state = sm.read() # returns tuple: (dx, dy, dz, rx, ry, rz)
#             if state is None:
#                 continue

#             dx, dy, dz, rx, ry, rz = state
#             print(f"Translation: dx={dx}, dy={dy}, dz={dz} | Rotation: rx={rx}, ry={ry}, rz={rz}")
#             print(f"Rotation: rx={rx}, ry={ry}, rz={rz}")

#             time.sleep(0.1)  # Polling interval
#     except KeyboardInterrupt:
#         print("Exiting Spacemouse reader.")
#         sm.close()


# if __name__ == "__main__":
#     main()


# # Minimal pytest test so pytest collects something and can skip gracefully when pyspacemouse isn't available.
# try:
#     import pytest  # type: ignore
# except Exception:
#     pytest = None  # type: ignore

# def test_import_pyspacemouse():
#     """Simple collection test: skip if pyspacemouse is not installed."""
#     if pytest is not None:
#         pytest.importorskip("pyspacemouse")
#     else:
#         # If running without pytest available, try import and silently skip on failure.
#         try:
#             import pyspacemouse  # type: ignore
#         except Exception:
#             return
#     assert True
