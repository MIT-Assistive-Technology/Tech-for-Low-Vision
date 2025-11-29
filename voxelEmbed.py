
import trimesh
import numpy as np
import scipy.ndimage as ndimage # <-- For 3D convolution
import cv2
import argparse
import sys
import time
from typing import List

epsilon = 1e-6
norm = lambda v: v / (np.linalg.norm(v) + epsilon)
pull = lambda f: trimesh.load(f, force="mesh")

def normalize_mesh(mesh):
    """Normalize mesh to be centered at origin and fit in a unit cube."""
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        return mesh
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    scale = (1.0 / mesh.scale) if mesh.scale > 0 else 1.0
    mesh.apply_scale(scale)
    return mesh

load = lambda f: normalize_mesh(pull(f))

def visualize_slices(slices: List[np.ndarray], axis: str, pitch: float):
    """
    Open an interactive OpenCV window to view slices.
    (Slightly modified to handle the new data)
    """
    if len(slices) == 0:
        print("No slices to visualize.", file=sys.stderr)
        return

    # Convert grayscale images to BGR so we can draw colored text
    # The images are already uint8, so this is simple
    try:
        slices_bgr = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in slices]
    except cv2.error as e:
        print(f"OpenCV error: {e}", file=sys.stderr)
        return

    total_slices = len(slices_bgr)
    current_index = 0
    window_name = f"Voxel Slice Viewer (Axis: {axis.upper()})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make window resizable

    print("\n--- 👁️ Voxel Viewer Active ---", file=sys.stderr)
    print(f"  Press 'n' or 'j' for NEXT slice", file=sys.stderr)
    print(f"  Press 'p' or 'k' for PREVIOUS slice", file=sys.stderr)
    print(f"  Press 'q' to QUIT", file=sys.stderr)
    print("-------------------------------", file=sys.stderr)

    while True:
        img_display = slices_bgr[current_index].copy()

        # Calculate real-world position
        position = (current_index - total_slices / 2) * pitch

        text = f"Slice: {current_index + 1} / {total_slices}"
        pos_text = f"Pos ({axis}): {position:.4f}"

        # cv2.putText(img_display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 2)
        # cv2.putText(img_display, pos_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 2)

        cv2.imshow(window_name, img_display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n') or key == ord('j'): # Next
            current_index = min(current_index + 1, total_slices - 1)
        elif key == ord('p') or key == ord('k'): # Previous
            current_index = max(current_index - 1, 0)

    cv2.destroyAllWindows()
def main():
    parser = argparse.ArgumentParser(description="Voxelize, convolve, and slice a 3D model.")

    parser.add_argument("filename", type=str, help="Model file (e.g., 'bunny.stl')")
    parser.add_argument("--pitch", type=float, default=0.01,
                        help="Size of each voxel (smaller = higher res). Default: 0.01")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Size of the 3D convolution kernel (e.g., 3 for 3x3x3). Default: 3")
    parser.add_argument("--axis", type=str, default='z', choices=['x', 'y', 'z'],
                        help="Axis to slice along (x, y, or z). Default: z")

    args = parser.parse_args()

    # --- 1. LOAD MESH ---
    print(f"Loading mesh: {args.filename}...", file=sys.stderr)
    mesh = load(args.filename)
    if not mesh.is_watertight:
        print("WARNING: Mesh is not watertight. Voxelization may be inaccurate.", file=sys.stderr)
        mesh.fill_holes()

    # --- 2. VOXELIZE ---
    print(f"Voxelizing with pitch={args.pitch}...", file=sys.stderr)
    start_time = time.perf_counter()

    voxel_grid = mesh.voxelized(pitch=args.pitch)
    voxel_matrix = voxel_grid.matrix.astype(float)

    print(f"Voxelization complete: {voxel_matrix.shape} grid", file=sys.stderr)
    print(f"--- Voxelization took {time.perf_counter() - start_time:.4f} seconds ---", file=sys.stderr)

    # --- 3. 3D CONVOLUTION ---
    k = args.kernel_size
    print(f"Performing 3D convolution with {k}x{k}x{k} kernel...", file=sys.stderr)
    start_time = time.perf_counter()

    kernel = np.ones((k, k, k)) / (k**3)
    convolved_matrix = ndimage.convolve(voxel_matrix, kernel)

    # --- THIS IS THE NEW LINE ---
    # Apply the original matrix as a mask to remove blur from the "air"
    convolved_matrix = convolved_matrix * voxel_matrix
    # --- END NEW LINE ---

    print(f"--- Convolution took {time.perf_counter() - start_time:.4f} seconds ---", file=sys.stderr)

    # --- 4. PREPARE SLICES ---
    print("Extracting and normalizing slices...", file=sys.stderr)
    slices = []

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_map.get(args.axis, 2)

    num_slices = convolved_matrix.shape[axis_index]

    grid_min = convolved_matrix.min()
    grid_max = convolved_matrix.max()
    normalized_grid = (convolved_matrix - grid_min) / (grid_max - grid_min + epsilon)
    image_grid = (normalized_grid * 255).astype(np.uint8)

    for i in range(num_slices):
        if args.axis == 'x':
            slice_img = image_grid[i, :, :]
        elif args.axis == 'y':
            slice_img = image_grid[:, i, :]
        else: # 'z'
            slice_img = image_grid[:, :, i]

        slices.append(slice_img)

    # --- 5. VISUALIZE ---
    visualize_slices(slices, args.axis, args.pitch)


if __name__ == "__main__":
    main()
