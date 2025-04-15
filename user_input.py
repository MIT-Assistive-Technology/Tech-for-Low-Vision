import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import math
import slicer
import time


# Helper Functions
######################


# may no longer use this first helper function , it actually seems easier to specify direction with a nit vector in mind
def string_to_unit_vector(rotation_string):
    # Split the string into three components and convert them to floats
    degrees = list(map(float, rotation_string.split(",")))

    # Convert degrees to radians
    radians = [math.radians(angle) for angle in degrees]

    # Calculate the vector magnitude
    magnitude = math.sqrt(sum(angle**2 for angle in radians))

    # Normalize the vector to create a unit vector
    unit_vector = [angle / magnitude for angle in radians]

    return unit_vector


def string_to_list(number_string):
    # Split the string by commas and convert each part to a float
    number_list = list(map(float, number_string.split(",")))
    return number_list


# User Defined Parameters and Slicing
###########################################################

# prompt user for filepath of stl
print("Type in filepath of STL:")
path = input()

mesh = trimesh.load_mesh(path)
while True:
    # prompt user for camera position
    print("Camera Position? (comma separated: x, y, z)")
    cam_pos_str = input()
    cam_pos_processed = string_to_list(cam_pos_str)
    camera_position = np.array(cam_pos_processed)

    # Now prompt user for camera direction
    print("Camera Direction? (comma separated unit vector e.g. 0, 0, -1)")
    cam_dir_str = input()

    cam_dir_vector = string_to_list(cam_dir_str)
    camera_direction = np.array(cam_dir_vector)

    # gather num slices from user
    # Print the initial message
    print("How many slices?")

    # Wait for the user to input a number and store it in a variable
    num_slices = int(input())

    # Print the status message
    print("Slicing...")

    start = time.time()
    # call slice_mesh again with specified number of slices
    slicer.slice_mesh(
        mesh, num_slices, camera_pos=camera_position, camera_dir=camera_direction
    )

    # plane_origins = [
    #     camera_position + i * 0.2 * camera_direction for i in range(num_slices)
    # ]
    # plane_normals = [camera_direction] * num_slices
    # slicer.plot_planes_with_mesh(mesh, plane_origins, plane_normals)

    end = time.time()

    print(f"elapsed time: {(end-start)/60} minutes")
