import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def points_to_image(points, resolution=(256, 256)):
	"""Convert an x by 2 matrix into an image matrix."""
	if len(points) == 0:
		return np.zeros(resolution)

	points = np.array(points)
	min_vals = points.min(axis=0)
	max_vals = points.max(axis=0)

	# Normalize points to fit in the image grid
	normalized_points = (points - min_vals) / (max_vals - min_vals + 1e-6)
	pixel_coords = (normalized_points * (np.array(resolution) - 1)).astype(int)

	image = np.zeros(resolution)
	for x, y in pixel_coords:
		image[y, x] = 1  # Mark points in the image matrix

	return image


def normalize_mesh(mesh):
	"""Normalize mesh to be centered at origin."""
	centroid = mesh.centroid
	scale = np.max(mesh.extents)
	mesh.apply_translation(-centroid)
	mesh.apply_scale(1.0 / scale)
	return mesh


def slice_mesh(mesh, num_slices, camera_pos, camera_dir):
	"""Slice a mesh based on an arbitrary camera position and direction."""
	mesh = normalize_mesh(mesh)
	camera_dir = np.array(camera_dir) / np.linalg.norm(
		camera_dir
	)  # Normalize direction

	bounds = mesh.bounds
	min_proj = np.dot(bounds[0], camera_dir)
	max_proj = np.dot(bounds[1], camera_dir)
	planes = np.linspace(min_proj, max_proj, num_slices)
	print(planes)

	for i, d in enumerate(planes):
		plane_origin = camera_pos + d * camera_dir
		slice_mesh = mesh.section(plane_origin=plane_origin, plane_normal=camera_dir)
		print("NOOGLE")
		if slice_mesh:
			slice_2D = slice_mesh.to_2D()[0]
			print("GOOGLE")
			points = slice_2D.vertices[:, :2]
			# plot points on a 2D plane in matplotlib
			plt.scatter(points[:, 0], points[:, 1])
			plt.title(f"Slice {i+1}/{num_slices}")
			plt.axis("off")
			plt.show(block=True)

			# image_matrix = points_to_image(points, resolution=(60, 60))

			# plt.imshow(image_matrix, cmap="gray")
			# plt.title(f"Slice {i+1}/{num_slices}")
			# plt.axis("off")
			# plt.show(block=True)


def plot_planes_with_mesh(mesh, plane_origins, plane_normals):
	"""Plot the mesh and a set of slicing planes in 3D."""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	# normalize mesh
	mesh = normalize_mesh(mesh)
 
	# Plot mesh
	ax.plot_trisurf(
		mesh.vertices[:, 0],
		mesh.vertices[:, 1],
		mesh.vertices[:, 2],
		triangles=mesh.faces,
		alpha=0.5,
		color="gray",
	)

	# Plot planes
	for origin, normal in zip(plane_origins, plane_normals):
		d = -np.dot(origin, normal)
		xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
		zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
		ax.plot_surface(xx, yy, zz, alpha=0.3, color="blue")

	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	plt.show()


# Load the model
mesh = trimesh.load_mesh("./tests/Benchmark_v.3.STL")
camera_position = np.array([0, 0, 1])
camera_direction = np.array([0, 0, -1])
# slice_mesh(
# 	mesh, num_slices=100, camera_pos=camera_position, camera_dir=camera_direction
# )


# Define example planes and plot
num_slices = 10
plane_origins = [
	camera_position + i * 0.2 * camera_direction for i in range(num_slices)
]
plane_normals = [camera_direction] * num_slices
plot_planes_with_mesh(mesh, plane_origins, plane_normals)
