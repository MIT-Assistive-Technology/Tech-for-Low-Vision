import hashlib
import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw

compose = lambda f, g: lambda x: f(g(x))
empty = lambda l: len(l) == 0
lerp = lambda t: lambda a: lambda b: (b - a) * t + a
epsilon = 1e-6  # avoid division by zero when normalizing a range of zero
max_slices = 100  # maximuim slice precision for caching
max_cache = 100  # maximum number of viewpoints to cache
normalize = lambda grid: lambda min: lambda max: (grid - min) / (max - min + epsilon)
norm = lambda v: v / np.linalg.norm(v)
pull = lambda f: trimesh.load(f, force="mesh")
sha256 = lambda s: hashlib.sha256(s).hexdigest()
hashify = compose(sha256, pickle.dumps)


def points_to_image(
    points: np.ndarray, resolution: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Convert an x by 2 matrix into an image matrix."""
    # Guard condition
    if empty(points):
        return np.zeros(resolution)

    points = np.array(points)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    # Normalize points to fit in the image grid
    normalized_points = normalize(points)(min_vals)(max_vals)
    pixel_coords = (normalized_points * (np.array(resolution) - 1)).astype(int)

    image = np.zeros(resolution)
    ones = np.ones(pixel_coords.shape[0])
    image[pixel_coords[:, 0], pixel_coords[:, 1]] = (
        ones  # mark points in the image matrix
    )

    return image


def normalize_mesh(mesh):
    """Normalize mesh to be centered at origin."""
    centroid = mesh.centroid
    scale = np.max(mesh.extents)
    mesh.apply_translation(-centroid)
    mesh.apply_scale(1.0 / scale)
    return mesh


def path2d_to_image(path, width=60, height=40) -> np.ndarray:
    # Get the bounds of the path
    minx, miny = path.bounds[0]
    maxx, maxy = path.bounds[1]

    # Compute scale and translation to fit path in the image
    scale_x = width / (maxx - minx)
    scale_y = height / (maxy - miny)
    scale = min(
        scale_x, scale_y
    )  # Preserve aspect ratio (use same scale in all directions)

    # Translate and scale vertices to image coordinates
    transformed = (path.vertices - [minx, miny]) * scale
    transformed[:, 1] = height - transformed[:, 1]  # Flip Y-axis for image coordinates

    # Create a blank image
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # Draw each path entity
    for entity in path.entities:
        if hasattr(entity, "points"):
            indices = entity.points
        elif hasattr(entity, "nodes"):
            indices = entity.nodes
        else:
            continue

        coords = [tuple(transformed[i]) for i in indices]
        if isinstance(entity, trimesh.path.entities.Line):  # type: ignore[reportAttributeAccessIssue]
            draw.line(coords, fill=255)
        elif isinstance(entity, trimesh.path.entities.Arc):  # type: ignore[reportAttributeAccessIssue]
            # Optional: Handle arc approximation
            arc = path.discrete(entity)
            coords = [(p[0], height - p[1]) for p in arc * scale]
            draw.line(coords, fill=255)

    return np.array(img)


def slice_mesh(
    mesh: trimesh.Trimesh,
    num_slices: int,
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
    disp=True,
):
    # get closest and furthest distances from the camera to the mesh given the camera position and direction

    # shoot a ray from the camera position in the direction of the camera direction and figure out the distance
    # at which it intersects with the mesh and the last distance it intersects with it
    # Use the ray-mesh intersector
    intersections, _, _ = mesh.ray.intersects_location(
        ray_origins=camera_pos,
        ray_directions=camera_dir,
        multiple_hits=True,
    )

    first_intersect = intersections[0, :]
    last_intersect = intersections[1, :]

    # get the first and last distances
    first_distances = np.linalg.norm(first_intersect - camera_pos)
    last_distances = np.linalg.norm(last_intersect - camera_pos)

    plane_origins = []

    # generate x plane origin coordinates in the same direction of the camera except all of their distances must range from first_distance to last_distance
    # for each slice, generate a plane origin at the camera direction
    for i in range(num_slices):
        # generate a plane origin at the camera direction
        plane_origins.append(
            camera_pos
            + lerp(i / num_slices)(first_distances)(last_distances) * camera_dir
        )

    # calculate the intersection of each one of the plane origin given they're all pointing in the camera dir
    # for each slice, generate a plane normal at the camera direction
    plane_normals = np.array(num_slices * [camera_dir])
    plane_origins = np.array(plane_origins)

    # get the intersection of the mesh with each one of the planes
    intersections = []

    # plot the mesh and all planes
    if disp:
        plot_planes_with_mesh(mesh, plane_origins, plane_normals)

    for i in range(num_slices):
        intersection_path = find_2d_intersection(
            plane_origin=plane_origins[i].reshape((3,)),
            plane_normal=plane_normals[i].reshape((3,)),
            mesh=mesh,
        )

        if intersection_path is not None:
            # convert the path to an image
            image = path2d_to_image(intersection_path, width=256, height=256)
            # append the image to the list of images
            intersections.append(image)

    return intersections


def find_2d_intersection(
    mesh: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray
):
    # Compute the intersection (section) of the plane and the mesh
    # This returns a Path3D object representing the intersection curve(s)
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    # Check if intersection was found
    if section is not None:
        # You can project it to 2D (if desired), or work with 3D paths
        slice_2D, _ = section.to_2D()
        return slice_2D
    else:
        print("Empty cross section")


def save_slice(image, path: str):
    """Save single slice image"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    plt.savefig(f"{path}.png", bbox_inches="tight", pad_inches=0)


def plot_slices(images):
    """Plot the images in a grid."""
    num_slices = len(images)
    cols = 5
    rows = (num_slices + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    for i in range(num_slices):
        ax = axs[i // cols, i % cols]
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    plt.show()


def plot_planes_with_mesh(mesh, plane_origins, plane_normals):
    """Plot the mesh and a set of slicing planes in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # normalize mesh
    mesh = normalize_mesh(mesh)

    # Plot mesh
    ax.plot_trisurf(  # type: ignore[reportAttributeAccessIssue]
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.5,
        color="gray",
    )

    # Plot planes
    for point, normal in zip(plane_origins, plane_normals):
        point = point.reshape((3,))
        normal = normal.reshape((3,))

        # Find two orthonormal vectors spanning the plane
        # First vector: arbitrary vector not collinear with the normal
        arbitrary = (
            np.array([1, 0, 0])
            if not np.allclose(normal[:2], [0, 0])
            else np.array([0, 1, 0])
        )
        v1 = np.cross(normal, arbitrary)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2)

        # Create a grid in plane coordinates
        s = np.linspace(-1, 1, 10)
        t = np.linspace(-1, 1, 10)
        S, T = np.meshgrid(s, t)

        # Generate points on the plane: P(s, t) = origin + s*v1 + t*v2
        plane_points = (
            point[:, None, None]
            + S[None, :, :] * v1[:, None, None]
            + T[None, :, :] * v2[:, None, None]
        )

        X = plane_points[0]
        Y = plane_points[1]
        Z = plane_points[2]

        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.5, color="blue")  # type: ignore[reportAttributeAccessIssue]

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore[reportAttributeAccessIssue]
    plt.show()


global_cache = "cache"
get_cache = lambda n: f"{global_cache}/{n}"
cache_entry = lambda entry: hashify(
    {"file": entry[0], "pose": entry[1], "dir": entry[2]}
)


# create cache directory for slices from specific viewpoint
def cache(slices: list, dirname: str, n: int):
    cachedir = get_cache(dirname)
    os.makedirs(cachedir, exist_ok=True)
    for i, slice in enumerate(slices):
        save_slice(slice, f"{cachedir}/{int(i / n * max_slices)}")


# normalize mesh and center it around 0,0 after pulling it from file source
load = compose(normalize_mesh, pull)


# generate cross sections given variable parameters, then cache images
def generate(file: str, pose: np.ndarray, dir: np.ndarray, n: int):
    mesh: trimesh.Trimesh = load(file)
    angle = norm(dir)
    slices = slice_mesh(
        mesh, num_slices=n, camera_pos=pose, camera_dir=angle, disp=False
    )
    cache(slices, cache_entry((file, pose, dir)), n)


# search cache for specific file
def search_cache_file(dir: str, p: float) -> str | None:
    id = int(p * max_slices)
    path = f"{dir}/{id}.png"
    return path if os.path.exists(path) else None


# search cache
def search_cache(file: str, pose: np.ndarray, dir: np.ndarray, p: float) -> str | None:
    dirname = cache_entry((file, pose, dir))
    cachedir = get_cache(dirname)
    return search_cache_file(cachedir, p) if os.path.isdir(cachedir) else None


# retrieve pre-generated slices or generate a new batch if necessary
def retrieve(file: str, pose: np.ndarray, dir: np.ndarray, n: int, i: int) -> str:
    res = search_cache(file, pose, dir, i / n)
    if res is not None:
        return res
    else:
        generate(file, pose, dir, n)
        return retrieve(file, pose, dir, n, i)


def test():
    print("Called from server")


def prune(subdirs: list[Path]):
    subdirs.sort(key=lambda p: p.stat().st_ctime)
    targeted = subdirs[max_cache:]
    for target in targeted:
        shutil.rmtree(target)


# clean cache by deleting old directories if exceeding max number of viewpoints
def clean():
    subdirs = [p for p in Path(global_cache).iterdir() if p.is_dir()]
    if len(subdirs) > max_cache:
        prune(subdirs)


if __name__ == "__main__":
    # Load the model
    mesh = load("./tests/mug.glb")
    camera_position = np.array([[0, 5, 0]])
    # find the direction that points to the origin
    camera_direction = 0 - camera_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    slices = slice_mesh(
        mesh,  # type: ignore[reportArgumentType]
        num_slices=10,
        camera_pos=camera_position,
        camera_dir=camera_direction,
    )
    plot_slices(slices)
