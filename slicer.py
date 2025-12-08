#!/usr/bin/env python3
"""
slicer.py — preserved feature set with global-consistent scaling

Features preserved:
- CLI: retrieve, visualize, clean (argument order unchanged)
- Caching in serve/assets/cache
- Timing decorator
- OpenCV-based viewer
- Multiprocessing start method preserved (but slicing runs sequentially to avoid pickling trimesh)
- Two slice modes: intersection (default) and full-projection (enable with --projection argument)
- Global consistent scale across all slices for any camera direction

Usage examples (same as before, optionally add --projection to get CT-like full silhouette slices):
    python slicer.py visualize mug.glb 0 0 2 0 0 -1 100
    python slicer.py visualize mug.glb 0 0 2 0 0 -1 100 --projection
    python slicer.py retrieve mug.glb 0 0 2 0 0 -1 100 5 --projection
"""

import hashlib
import os
import pickle
import shutil
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Any
import time
import functools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage

import cv2
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# shapely is optional but recommended for robust polygon unions
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

# -------------------------
# Config / Defaults
# -------------------------
epsilon = 1e-9
max_slices = 100
max_cache = 100

# By default preserve intersection-based original behavior.
# Pass '--projection' on the command line to enable full-object-projection (CT-like) slices.
DEFAULT_FULL_PROJECTION = False

# Output image size
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256

# -------------------------
# Utilities
# -------------------------
def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"--- Timing '{func.__name__}': Starting ---", file=sys.stderr)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"--- Timing '{func.__name__}': Finished in {elapsed:.4f} seconds ---", file=sys.stderr)
        return result
    return wrapper

compose = lambda f, g: lambda x: f(g(x))
empty = lambda l: len(l) == 0
norm = lambda v: v / (np.linalg.norm(v) + epsilon)
pull = lambda f: trimesh.load(f, force="mesh")
sha256 = lambda s: hashlib.sha256(s).hexdigest()
hashify = compose(sha256, pickle.dumps)

def lerp(a, b, t):
    return (b - a) * t + a

def normalize_array(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val + epsilon)

# -------------------------
# Caching
# -------------------------
global_cache = "serve/assets/cache"
get_cache = lambda n: os.path.join(global_cache, n)

def cache_entry(file: str, pose: np.ndarray, direction: np.ndarray, n: int, full_projection: bool) -> str:
    """Generate cache key for a specific viewpoint & mode."""
    data_to_hash = (file, tuple(pose.flatten()), tuple(direction.flatten()), int(n), bool(full_projection))
    return hashify(data_to_hash)

# -------------------------
# IO helpers
# -------------------------
def save_slice(image_array: np.ndarray, path: str):
    if image_array is None:
        return
    img = image_array.astype(np.uint8)
    Image.fromarray(img, 'L').save(f"{path}.png")

# -------------------------
# Geometry helpers (global-consistent scaling)
# -------------------------
def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a stable orthonormal basis (x_axis, y_axis, normal) for a plane whose normal is 'normal'.
    """
    n = np.array(normal, dtype=float).flatten()
    n /= (np.linalg.norm(n) + epsilon)

    # choose an arbitrary reference not parallel to n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(n, ref)
    x_axis /= (np.linalg.norm(x_axis) + epsilon)
    y_axis = np.cross(n, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + epsilon)

    return x_axis, y_axis, n

def project_vertices_to_plane(vertices: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """
    Project 3D vertices into 2D coordinates using x_axis and y_axis.
    Returns shape (N,2).
    """
    v_x = vertices @ x_axis
    v_y = vertices @ y_axis
    return np.vstack([v_x, v_y]).T

def compute_global_2d_bounds(mesh: trimesh.Trimesh, plane_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given plane normal (camera_dir normalized), compute a global 2D bounding box by projecting
    the entire mesh's vertices into the derived plane basis. Returns (global_min, global_max, basis_axes)
    where global_min/max are 2-vectors and basis_axes = (x_axis, y_axis, normal).
    """
    x_axis, y_axis, n = build_plane_basis(plane_normal)
    verts_2d = project_vertices_to_plane(mesh.vertices, x_axis, y_axis)
    global_min = verts_2d.min(axis=0)
    global_max = verts_2d.max(axis=0)
    return global_min, global_max, (x_axis, y_axis, n)

def compute_global_scale(global_min: np.ndarray, global_max: np.ndarray, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    W = float(global_max[0] - global_min[0])
    H = float(global_max[1] - global_min[1])
    if W < epsilon or H < epsilon:
        return 1.0, 0.0, 0.0
    scale = 0.9 * min(width / W, height / H)
    offset_x = (width - W * scale) / 2.0
    offset_y = (height - H * scale) / 2.0
    return scale, offset_x, offset_y

# -------------------------
# Polygon generation (intersection and full-projection)
# -------------------------
def polygon_from_section(section, plane_normal) -> Optional[Any]:
    """
    Convert a trimesh section to a shapely (or equivalent) polygon in the consistent 2D frame.
    Returns a shapely geometry (Polygon/MultiPolygon) if shapely available, otherwise returns None.
    """
    if section is None:
        return None

    try:
        # build consistent to_2D transform matching build_plane_basis
        normal = np.array(plane_normal, dtype=float).flatten()
        if abs(normal[2]) < 0.9:
            reference = np.array([0.0, 0.0, 1.0])
        else:
            reference = np.array([1.0, 0.0, 0.0])

        x_axis = np.cross(normal, reference)
        x_axis /= (np.linalg.norm(x_axis) + epsilon)
        y_axis = np.cross(normal, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + epsilon)

        to_2D_transform = np.eye(4)
        to_2D_transform[:3, 0] = x_axis
        to_2D_transform[:3, 1] = y_axis
        to_2D_transform[:3, 2] = normal

        out = section.to_2D(to_2D_transform)
        if isinstance(out, tuple) and len(out) > 0:
            path2d = out[0]
        else:
            path2d = out

        if path2d is None:
            return None

        if SHAPELY_AVAILABLE:
            polys = []
            if hasattr(path2d, "polygons_full"):
                for p in path2d.polygons_full:
                    try:
                        exterior = list(p.exterior.coords)
                        interiors = [list(i.coords) for i in p.interiors] if hasattr(p, "interiors") else []
                        polys.append(Polygon(exterior, interiors))
                    except Exception:
                        continue
            if not polys:
                return None
            return unary_union(polys)
        else:
            # Fallback: return the trimesh Path2D object for rasterization using path.polygons_full later.
            return path2d

    except Exception as e:
        print(f"[polygon_from_section] failed: {e}", file=sys.stderr)
        return None

def polygon_from_full_projection(mesh: trimesh.Trimesh, plane_normal) -> Optional[Any]:
    """
    Project all triangles into the plane basis and union them to obtain the full silhouette polygon.
    Returns shapely geometry if available, otherwise returns None or a list of polygons.
    """
    x_axis, y_axis, _ = build_plane_basis(plane_normal)
    verts_2d = project_vertices_to_plane(mesh.vertices, x_axis, y_axis)

    polys = []
    for face in mesh.faces:
        pts = verts_2d[face]
        try:
            p = Polygon(pts)
            if p.is_valid and p.area > epsilon:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    if SHAPELY_AVAILABLE:
        merged = unary_union(polys)
        return merged
    else:
        # fallback: return list of shapely-like polygons as plain arrays
        return polys

# -------------------------
# Rasterization using global scale
# -------------------------
def render_shapely_or_path_to_image(poly, global_min: np.ndarray, scale: float, offset_x: float, offset_y: float, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """
    Poly can be:
      - shapely Polygon / MultiPolygon
      - trimesh Path2D (if shapely not present)
      - list of triangle Polygons (fallback)
    """
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    if poly is None:
        return np.array(img, dtype=np.uint8)

    if SHAPELY_AVAILABLE and (isinstance(poly, Polygon) or isinstance(poly, MultiPolygon)):
        poly_list = [poly] if isinstance(poly, Polygon) else list(poly)
        for p in poly_list:
            try:
                exterior = np.array(p.exterior.coords)
                ext = (exterior - global_min) * scale
                ext += np.array([offset_x, offset_y])
                ext[:, 1] = height - ext[:, 1]
                draw.polygon([tuple(pt) for pt in ext], fill=255)
                for interior in p.interiors:
                    h = np.array(interior.coords)
                    h = (h - global_min) * scale
                    h += np.array([offset_x, offset_y])
                    h[:, 1] = height - h[:, 1]
                    draw.polygon([tuple(pt) for pt in h], fill=0)
            except Exception as e:
                print(f"[render] shapely polygon error: {e}", file=sys.stderr)
                continue
        return np.array(img, dtype=np.uint8)

    # If we don't have shapely, attempt to rasterize trimesh Path2D or raw triangles
    if hasattr(poly, "polygons_full"):
        try:
            for p in poly.polygons_full:
                ext_coords = np.array(p.exterior.coords)
                ext = (ext_coords - global_min) * scale
                ext += np.array([offset_x, offset_y])
                ext[:, 1] = height - ext[:, 1]
                draw.polygon([tuple(pt) for pt in ext], fill=255)
                for interior in getattr(p, "interiors", []):
                    h = np.array(interior.coords)
                    h = (h - global_min) * scale
                    h += np.array([offset_x, offset_y])
                    h[:, 1] = height - h[:, 1]
                    draw.polygon([tuple(pt) for pt in h], fill=0)
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"[render] Path2D rasterization failed: {e}", file=sys.stderr)

    # fallback: if poly is a list of triangle-polygons (as numpy arrays)
    if isinstance(poly, list):
        for p in poly:
            try:
                pts = np.array(p.exterior.coords)
                ext = (pts - global_min) * scale
                ext += np.array([offset_x, offset_y])
                ext[:, 1] = height - ext[:, 1]
                draw.polygon([tuple(pt) for pt in ext], fill=255)
            except Exception:
                continue
        return np.array(img, dtype=np.uint8)

    return np.array(img, dtype=np.uint8)

# -------------------------
# Ray bounds helper
# -------------------------
def get_mesh_bounds_along_ray(mesh: trimesh.Trimesh, camera_pos: np.ndarray, camera_dir: np.ndarray) -> Tuple[float, float]:
    dir_norm = np.array(camera_dir, dtype=float).flatten()
    dir_norm /= (np.linalg.norm(dir_norm) + epsilon)

    intersections, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([camera_pos], dtype=float),
        ray_directions=np.array([dir_norm], dtype=float)
    )

    if len(intersections) >= 2:
        distances = np.dot(intersections - camera_pos, dir_norm)
        return float(distances.min()), float(distances.max())

    # fallback: use bounding box corners projection
    corners = mesh.bounding_box.vertices
    if len(corners) == 0:
        raise ValueError("Cannot determine mesh bounds along the ray.")
    projections = np.dot(corners - camera_pos, dir_norm)
    return float(projections.min()), float(projections.max())

# -------------------------
# Main slice function (maintains global scale)
# -------------------------
@timeit
def slice_mesh(
    mesh: trimesh.Trimesh,
    num_slices: int,
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    full_projection: bool = DEFAULT_FULL_PROJECTION,
) -> list:
    """
    Generate slices. This function ensures global-consistent scale across slices:
      - a 2D basis is derived from camera_dir (normalized)
      - the entire mesh is projected into that basis to compute global bounds
      - every slice uses that same scale + offsets
    If full_projection is True, each slice contains the full silhouette obtained by projecting
    every triangle. If False, we use mesh.section intersections (original behavior).
    """
    dir_norm = np.array(camera_dir, dtype=float).flatten()
    dir_norm /= (np.linalg.norm(dir_norm) + epsilon)

    try:
        first_distance, last_distance = get_mesh_bounds_along_ray(mesh, camera_pos, dir_norm)
    except ValueError as e:
        print(f"Error getting mesh bounds: {e}", file=sys.stderr)
        return []

    near = float(min(first_distance, last_distance))
    far = float(max(first_distance, last_distance))

    distances = np.linspace(near, far, num_slices)
    plane_origins = [camera_pos + d * dir_norm for d in distances]

    # Compute global 2D frame & bounds (same for all slices)
    global_min, global_max, basis = compute_global_2d_bounds(mesh, dir_norm)
    scale, offset_x, offset_y = compute_global_scale(global_min, global_max, width, height)

    results = []

    # Precompute full-projection polygon once if requested (it's independent of origin)
    full_proj_poly = None
    if full_projection:
        full_proj_poly = polygon_from_full_projection(mesh, dir_norm)

    for idx, origin in enumerate(plane_origins):
        poly = None
        if full_projection:
            poly = full_proj_poly
        else:
            # Intersection-based section at this plane origin
            try:
                section = mesh.section(plane_origin=origin, plane_normal=dir_norm)
                poly = polygon_from_section(section, dir_norm)
            except Exception as e:
                print(f"[slice_mesh] section failed at idx {idx}: {e}", file=sys.stderr)
                poly = None

        img = render_shapely_or_path_to_image(poly, global_min, scale, offset_x, offset_y, width, height)
        results.append(img)

    return results

# -------------------------
# Normalization & loader
# -------------------------
def normalize_mesh(mesh):
    """Normalize mesh to be centered at origin and fit in a unit cube."""
    if not isinstance(mesh, trimesh.Trimesh) or empty(mesh.vertices):
        return mesh
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    scale_val = (1.0 / mesh.scale) if mesh.scale > 0 else 1.0
    mesh.apply_scale(scale_val)
    return mesh

load = compose(normalize_mesh, pull)

# -------------------------
# Caching helpers
# -------------------------
@timeit
def cache_slices(slices: list, dirname: str):
    cachedir = get_cache(dirname)
    os.makedirs(cachedir, exist_ok=True)
    for i, slice_data in enumerate(slices):
        save_slice(slice_data, os.path.join(cachedir, str(i+1)))

@timeit
def generate_and_cache(file: str, pose: np.ndarray, direction: np.ndarray, n: int, cache_key: str, full_projection: bool = DEFAULT_FULL_PROJECTION):
    try:
        mesh: trimesh.Trimesh = load(file)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load a valid mesh from {file}")
    except Exception as e:
        print(f"Error loading mesh: {e}", file=sys.stderr)
        return

    angle = norm(direction)
    slices = slice_mesh(mesh, num_slices=n, camera_pos=pose, camera_dir=angle, full_projection=full_projection)

    if not empty(slices):
        cache_slices(slices, cache_key)

def retrieve(file: str, pose: np.ndarray, direction: np.ndarray, n: int, i: int, full_projection: bool = DEFAULT_FULL_PROJECTION) -> dict:
    cache_key = cache_entry(file, pose, direction, n, full_projection)
    cachedir = get_cache(cache_key)

    slice_path = os.path.join(cachedir, f"{i}.png")

    if not os.path.isdir(cachedir):
        print(f"Cache miss. Generating {n} slices for {cache_key} (full_projection={full_projection})...", file=sys.stderr)
        generate_and_cache(file, pose, direction, n, cache_key, full_projection)

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

# -------------------------
# Clean
# -------------------------
def clean():
    cache_dir = Path("serve") / "assets" / "cache"

    if cache_dir.exists() and cache_dir.is_dir():
        try:
            shutil.rmtree(cache_dir)
            print(f"[CLEAN] Cache directory '{cache_dir}' successfully removed.", file=sys.stderr)
        except OSError as e:
            print(f"[ERROR] Could not remove cache directory {cache_dir}: {e}", file=sys.stderr)
            return

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CLEAN] Empty cache directory '{cache_dir}' re-created.", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Could not re-create cache directory: {e}", file=sys.stderr)

# -------------------------
# Viewer
# -------------------------
def visualize_slices(slices: list):
    if empty(slices):
        print("No slices to visualize.", file=sys.stderr)
        return

    try:
        slices_bgr = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR) for img in slices]
    except cv2.error as e:
        print(f"OpenCV error. Make sure all slices are valid images. {e}", file=sys.stderr)
        return

    total_slices = len(slices_bgr)
    current_index = 0
    window_name = "Slice Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- 👁️ Slice Viewer Active ---", file=sys.stderr)
    print(f"  Press 'n' or 'j' for NEXT slice", file=sys.stderr)
    print(f"  Press 'p' or 'k' for PREVIOUS slice", file=sys.stderr)
    print(f"  Press 'q' to QUIT", file=sys.stderr)
    print("-------------------------------", file=sys.stderr)

    while True:
        img_display = slices_bgr[current_index].copy()
        text = f"Slice: {current_index + 1} / {total_slices}"
        cv2.putText(img_display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n') or key == ord('j'):
            current_index = min(current_index + 1, total_slices - 1)
        elif key == ord('p') or key == ord('k'):
            current_index = max(current_index - 1, 0)

    cv2.destroyAllWindows()

# -------------------------
# Main CLI
# -------------------------
def parse_full_projection_flag(argv):
    return "--projection" in argv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided."}), file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    full_projection_flag = parse_full_projection_flag(sys.argv)

    try:
        if command == "retrieve":
            file_name = sys.argv[2]
            file_path = os.path.join("models", file_name)

            pose = np.array([float(x) for x in sys.argv[3:6]])
            direction = np.array([float(x) for x in sys.argv[6:9]])
            n_slices = int(sys.argv[9])
            i_slice = int(sys.argv[10])

            result = retrieve(file_path, pose, direction, n_slices, i_slice, full_projection=full_projection_flag)

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

            print(f"Generating {n_slices} slices in memory... (projection={full_projection_flag})", file=sys.stderr)
            angle = norm(direction)

            slices = slice_mesh(mesh, num_slices=n_slices, camera_pos=pose, camera_dir=angle, full_projection=full_projection_flag)

            visualize_slices(slices)

        elif command == "clean":
            clean()

        else:
            print(json.dumps({"error": f"Unknown command: {command}"}), file=sys.stderr)

    except (IndexError, ValueError) as e:
        print(json.dumps({"error": f"Invalid arguments for command '{command}'. Details: {e}"}), file=sys.stderr)
        sys.exit(1)
