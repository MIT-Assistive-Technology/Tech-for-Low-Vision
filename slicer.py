import hashlib
import os
import pickle
import shutil
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Any
import time # <-- NEW IMPORT (for timeit)
import functools # <-- NEW IMPORT (for partial and wraps)
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage

import cv2
#added multiprocessing
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

"""
how to use "visualize" command!!

example command: python slicer.py visualize mug.glb 0 0 2 0 0 -1 100
The script reads these numbers in a strict order:

0 0 2: This is the Pose (camera_pos).
It's an (x, y, z) coordinate of where the "camera" (or slicing-plane) starts.
In this case, it starts at (0, 0, 2), which is 2 units "above" the center of the normalized object.

0 0 -1: This is the Direction (camera_dir).
It's an (x, y, z) vector of where the camera is looking.

(0, 0, -1) means it's looking straight down the Z-axis.
100: This is the Number of Slices (n_slices).
"""

def timeit(func):
    """A simple decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Print to stderr so it doesn't interfere with JSON output
        print(f"--- Timing '{func.__name__}': Starting ---", file=sys.stderr)
        start_time = time.perf_counter()

        result = func(*args, **kwargs) # Run the actual function

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"--- Timing '{func.__name__}': Finished in {elapsed:.4f} seconds ---", file=sys.stderr)
        return result
    return wrapper
# --- END DECORATOR ---


# Simplified utility functions
compose = lambda f, g: lambda x: f(g(x))
empty = lambda l: len(l) == 0
epsilon = 1e-6
max_slices = 100
max_cache = 100
norm = lambda v: v / (np.linalg.norm(v) + epsilon)
pull = lambda f: trimesh.load(f, force="mesh")
sha256 = lambda s: hashlib.sha256(s).hexdigest()
hashify = compose(sha256, pickle.dumps)


def lerp(a, b, t):
    """Linear interpolation."""
    return (b - a) * t + a


def normalize_array(arr, min_val, max_val):
    """Normalize array to [0, 1] range."""
    return (arr - min_val) / (max_val - min_val + epsilon)

def path2d_to_image(path, width=256, height=256) -> np.ndarray:
    """Convert 2D path to filled image using PIL and shapely polygons."""
    if path is None or not hasattr(path, 'bounds') or empty(path.vertices):
        # Return an all-black image (representing empty "air")
        return np.zeros((width, height))

    minx, miny = path.bounds[0]
    maxx, maxy = path.bounds[1]

    # Handle cases where the path is a single point or line
    if abs(maxx - minx) < epsilon or abs(maxy - miny) < epsilon:
        return np.zeros((width, height)) # Return empty image

    scale_x = width / (maxx - minx)
    scale_y = height / (maxy - miny)
    scale = min(scale_x, scale_y) * 0.9 # Add some padding

    offset_x = (width - (maxx - minx) * scale) / 2
    offset_y = (height - (maxy - miny) * scale) / 2

    # Create a black background ("air")
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # --- THIS IS THE ONLY CHANGE ---
    # It should be .polygons_full, not .polygons
    for polygon in path.polygons_full:
    # --- END CHANGE ---

        # --- 1. Draw the exterior (solid) ---
        exterior_coords = np.array(polygon.exterior.coords)

        # Apply the same transformations as before
        transformed_ext = (exterior_coords - [minx, miny]) * scale
        transformed_ext += [offset_x, offset_y]
        transformed_ext[:, 1] = height - transformed_ext[:, 1] # Flip Y-axis

        # Draw the filled exterior shape as white ("solid")
        draw.polygon([tuple(p) for p in transformed_ext], fill=255)

        # --- 2. Draw the interiors (holes) ---
        for interior in polygon.interiors:
            interior_coords = np.array(interior.coords)

            # Apply the same transformations
            transformed_int = (interior_coords - [minx, miny]) * scale
            transformed_int += [offset_x, offset_y]
            transformed_int[:, 1] = height - transformed_int[:, 1] # Flip Y-axis

            # Draw the filled interior shape as black ("air")
            # This "punches the hole" in the white shape
            draw.polygon([tuple(p) for p in transformed_int], fill=0)

    return np.array(img)

def get_mesh_bounds_along_ray(mesh: trimesh.Trimesh, camera_pos: np.ndarray, camera_dir: np.ndarray) -> Tuple[float, float]:
    """Get min and max distances along a ray through the mesh."""
    intersections, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([camera_pos]), ray_directions=np.array([camera_dir])
    )

    if len(intersections) < 2:
        corners = mesh.bounding_box.vertices
        projections = np.dot(corners - camera_pos, camera_dir)
        if len(projections) == 0:
            raise ValueError("Cannot determine mesh bounds along the ray.")
        return float(projections.min()), float(projections.max())

    distances = np.linalg.norm(intersections - camera_pos, axis=1)
    return float(distances.min()), float(distances.max())

def _process_slice_worker(plane_origin, mesh, plane_normal):
    """
    Worker function for a single slice.
    Ensures consistent orientation by using a fixed transformation.
    """
    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        slice_2D = None

        if section is None:
            pass
        elif hasattr(section, 'to_2D'):
            if len(section.vertices) > 0:
                try:
                    # CRITICAL FIX: Specify a consistent transformation matrix
                    # Create orthonormal basis from plane_normal
                    normal = plane_normal.flatten()
                    
                    # Find two perpendicular vectors in the plane
                    # Choose a reference vector that's not parallel to normal
                    if abs(normal[2]) < 0.9:
                        reference = np.array([0, 0, 1])
                    else:
                        reference = np.array([1, 0, 0])
                    
                    # Create consistent basis vectors
                    x_axis = np.cross(normal, reference)
                    x_axis = x_axis / (np.linalg.norm(x_axis) + epsilon)
                    y_axis = np.cross(normal, x_axis)
                    y_axis = y_axis / (np.linalg.norm(y_axis) + epsilon)
                    
                    # Build transformation matrix
                    to_2D_transform = np.eye(4)
                    to_2D_transform[:3, 0] = x_axis
                    to_2D_transform[:3, 1] = y_axis
                    to_2D_transform[:3, 2] = normal
                    
                    # Apply the consistent transformation
                    slice_2D, _ = section.to_2D(to_2D_transform)
                    
                except Exception as e:
                    print(f"Slice at {plane_origin}: .to_2D() method failed: {e}", file=sys.stderr)
        elif hasattr(section, 'outline'):
            if len(section.vertices) > 0:
                slice_2D = section.outline()
        else:
            print(f"Slice at {plane_origin}: received unknown section type: {type(section)}", file=sys.stderr)

        image = path2d_to_image(slice_2D)
        return image

    except Exception as e:
        print(f"FATAL Error processing slice at {plane_origin}: {e}", file=sys.stderr)
        return np.zeros((256, 256), dtype=np.uint8)

@timeit # <-- APPLIED DECORATOR
def slice_mesh(
    mesh: trimesh.Trimesh,
    num_slices: int,
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
) -> list:
    """Slice mesh along camera direction using multiprocessing."""
    try:
        first_distance, last_distance = get_mesh_bounds_along_ray(mesh, camera_pos, camera_dir)
    except ValueError as e:
        print(f"Error getting mesh bounds: {e}", file=sys.stderr)
        return []

    t_values = np.linspace(0, 1, num_slices)
    plane_origins = camera_pos + np.outer(
        lerp(first_distance, last_distance, t_values),
        camera_dir.flatten()
    )

    task_worker = functools.partial(
        _process_slice_worker,
        mesh=mesh,
        plane_normal=camera_dir
    )

    # intersections = []

    # with ProcessPoolExecutor() as executor:
    #     results = executor.map(task_worker, plane_origins)
    #     intersections = list(results)
    # Trimesh object is not picklable, removed for now
    intersections = []
    for origin in plane_origins:
        intersections.append(task_worker(origin))


    return intersections


def normalize_mesh(mesh):
    """Normalize mesh to be centered at origin and fit in a unit cube."""
    if not isinstance(mesh, trimesh.Trimesh) or empty(mesh.vertices):
        return mesh
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    scale = (1.0 / mesh.scale) if mesh.scale > 0 else 1.0
    mesh.apply_scale(scale)
    return mesh


def save_slice(image_array: np.ndarray, path: str):
    if image_array is None: 
        return
    
    # Fix for blank images (ensures correct data type)
    img = image_array.astype(np.uint8) 
    
    Image.fromarray(img, 'L').save(f"{path}.png")


# --- Cache Management ---
global_cache = "serve/assets/cache"
get_cache = lambda n: os.path.join(global_cache, n)

def cache_entry(file: str, pose: np.ndarray, direction: np.ndarray) -> str:
    """Generate cache key for a specific viewpoint."""
    data_to_hash = (file, tuple(pose.flatten()), tuple(direction.flatten()))
    return hashify(data_to_hash)


@timeit # <-- APPLIED DECORATOR
def cache_slices(slices: list, dirname: str):
    """Create cache directory and save all slice images."""
    cachedir = get_cache(dirname)
    os.makedirs(cachedir, exist_ok=True)
    for i, slice_data in enumerate(slices):
        save_slice(slice_data, os.path.join(cachedir, str(i)))


load = compose(normalize_mesh, pull)

@timeit # <-- APPLIED DECORATOR
def generate_and_cache(file: str, pose: np.ndarray, directory: np.ndarray, n: int, cache_key: str):
    """Generate all cross sections, then cache them."""
    try:
        mesh: trimesh.Trimesh = load(file)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load a valid mesh from {file}")
    except Exception as e:
        print(f"Error loading mesh: {e}", file=sys.stderr)
        return

    angle = norm(directory)
    slices = slice_mesh(mesh, num_slices=n, camera_pos=pose, camera_dir=angle)

    if not empty(slices):
        cache_slices(slices, cache_key)

def retrieve(file: str, pose: np.ndarray, directory: np.ndarray, n: int, i: int) -> dict:
    """Retrieve a specific slice, generating the full set if necessary."""
    cache_key = cache_entry(file, pose, directory)
    cachedir = get_cache(cache_key)

    slice_path = os.path.join(cachedir, f"{i}.png")

    if not os.path.isdir(cachedir):
        print(f"Cache miss. Generating {n} slices for {cache_key}...", file=sys.stderr)

        # This function call is now timed by the decorator
        generate_and_cache(file, pose, directory, n, cache_key)

    if os.path.exists(slice_path):
        image = Image.open(slice_path)
        image_data = np.array(image).tolist()

        return {
            "path": slice_path,
            "sliceData": image_data,
            "width": image.width,
            "height": image.height,
            "message": "Slice retrieved successfully."
        }
    else:
        return {
            "error": f"Failed to generate or find slice {i} for the given parameters."
        }

# def clean():
#     """Clean cache by deleting old directories."""
#     cache_path = Path(global_cache)
#     if not cache_path.exists(): return

#     subdirs = [p for p in cache_path.iterdir() if p.is_dir()]
#     if len(subdirs) > max_cache:
#         subdirs.sort(key=lambda p: p.stat().st_ctime)
#         for target in subdirs[:len(subdirs) - max_cache]:
#             shutil.rmtree(target)
#     print(json.dumps({"message": "Cache cleaned successfully."}))
def clean():
    """Deletes the cache directory and recreates it."""
    # Define the path relative to the root where slicer.py is run
    cache_dir = Path("serve") / "assets" / "cache"
    
    # --- Deletion ---
    if cache_dir.exists() and cache_dir.is_dir():
        try:
            # Recursively deletes the directory and all files inside it
            shutil.rmtree(cache_dir)
            print(f"[CLEAN] Cache directory '{cache_dir}' successfully removed.", file=sys.stderr)
        except OSError as e:
            # Handle potential permissions issues
            print(f"[ERROR] Could not remove cache directory {cache_dir}: {e}", file=sys.stderr)
            return

    # --- Re-creation ---
    # The application needs this directory to exist to save new slices
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CLEAN] Empty cache directory '{cache_dir}' re-created.", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Could not re-create cache directory: {e}", file=sys.stderr)


def visualize_slices(slices: list):
    """
    Open an interactive OpenCV window to view slices.

    Controls:
        n / j : Next slice
        p / k : Previous slice
        q     : Quit
    """
    if empty(slices):
        print("No slices to visualize.", file=sys.stderr)
        return

    # Convert grayscale images to BGR so we can draw colored text
    try:
        slices_bgr = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR) for img in slices]
    except cv2.error as e:
        print(f"OpenCV error. Make sure all slices are valid images. {e}", file=sys.stderr)
        return

    total_slices = len(slices_bgr)
    current_index = 0
    window_name = "Slice Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make window resizable

    print("\n--- 👁️ Slice Viewer Active ---", file=sys.stderr)
    print(f"  Press 'n' or 'j' for NEXT slice", file=sys.stderr)
    print(f"  Press 'p' or 'k' for PREVIOUS slice", file=sys.stderr)
    print(f"  Press 'q' to QUIT", file=sys.stderr)
    print("-------------------------------", file=sys.stderr)

    while True:
        # Get the current slice and make a copy so we don't draw on it
        img_display = slices_bgr[current_index].copy()

        # Add text overlay
        text = f"Slice: {current_index + 1} / {total_slices}"
        cv2.putText(
            img=img_display,
            text=text,
            org=(10, 25), # Bottom-left corner of the text
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 255, 0), # Green
            thickness=2
        )

        cv2.imshow(window_name, img_display)

        # Wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n') or key == ord('j'): # Next
            current_index = min(current_index + 1, total_slices - 1)
        elif key == ord('p') or key == ord('k'): # Previous
            current_index = max(current_index - 1, 0)
        # Note: Arrow keys are often platform-specific, so 'n'/'p' is more robust

    cv2.destroyAllWindows()


# --- Main execution block for command-line calls ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided."}), file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "retrieve":
            file_name = sys.argv[2]
            file_path = os.path.join("models", file_name)

            pose = np.array([float(x) for x in sys.argv[3:6]])
            direction = np.array([float(x) for x in sys.argv[6:9]])
            n_slices = int(sys.argv[9])
            i_slice = int(sys.argv[10])

            result = retrieve(file_path, pose, direction, n_slices, i_slice)

            # The final JSON output is printed to stdout
            print(json.dumps(result))

        elif command == "visualize":
            file_name = sys.argv[2]
            file_path = os.path.join("models", file_name)

            pose = np.array([float(x) for x in sys.argv[3:6]])
            direction = np.array([float(x) for x in sys.argv[6:9]])
            n_slices = int(sys.argv[9])

            print(f"Loading mesh: {file_path}", file=sys.stderr)
            mesh: trimesh.Trimesh = load(file_path)
            if not isinstance(mesh, trimesh.Trimesh):
                 raise ValueError(f"Failed to load a valid mesh from {file_path}")

            print(f"Generating {n_slices} slices in memory...", file=sys.stderr)
            angle = norm(direction)

            # This will be timed by your @timeit decorator
            slices = slice_mesh(mesh, num_slices=n_slices, camera_pos=pose, camera_dir=angle)

            visualize_slices(slices)

        elif command == "clean":
            clean()

        else:
            print(json.dumps({"error": f"Unknown command: {command}"}), file=sys.stderr)

    except (IndexError, ValueError) as e:
        print(json.dumps({"error": f"Invalid arguments for command '{command}'. Details: {e}"}), file=sys.stderr)
        sys.exit(1)
