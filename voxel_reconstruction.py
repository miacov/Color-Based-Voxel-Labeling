import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import utils


def load_config_info(config_info_path="data/cam", config_input_filename="config.xml"):
    """
    Loads intrinsic (camera matrix, distortion coefficients) and extrinsic (rotation vector, translation vector) camera
    parameters from config file.

    :param config_info_path: config xml file directory path
    :param config_input_filename: config xml file name
    :return: camera matrix
    """
    # Select tags for loaded nodes and their types
    node_tags = ["CameraMatrix", "DistortionCoeffs", "RotationVector", "TranslationVector"]
    node_types = ["mat" for _ in range(len(node_tags))]

    # Load nodes
    nodes = utils.load_xml_nodes(config_info_path, config_input_filename, node_tags, node_types)

    # Parse config
    mtx = nodes.get("CameraMatrix")
    dist = nodes.get("DistortionCoeffs")
    rvecs = nodes.get("RotationVector")
    tvecs = nodes.get("TranslationVector")

    return mtx, dist, rvecs, tvecs


def create_voxel_volume(voxel_volume_shape, voxel_size, block_size):
    """
    Creates voxel volume points given dimensions.

    :param voxel_volume_shape: voxel volume shape dimensions (width, height, depth)
    :param voxel_size: distance in mm for voxels
    :param block_size: block size unit
    :return: returns voxel volume points (dimensions as width x depth x height, height being flipped)
    """
    # Volume shape dimensions
    width = voxel_volume_shape[0]
    height = voxel_volume_shape[1]
    depth = voxel_volume_shape[2]

    # Create volume space
    voxel_points = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # Swap y and z
                voxel_points.append(
                    [voxel_size * (x * block_size - width / 2),
                     voxel_size * (z * block_size - depth / 2),
                     - voxel_size * (y * block_size)])

    return np.array(voxel_points, dtype=np.float32)


def create_lookup_table(voxel_points, num_cameras, cam_input_path="data", config_input_filename="config.xml"):
    """
    Creates lookup table to map 3D voxels to 2D points for a number of cameras.

    :param voxel_points: voxel volume points (dimensions as width x depth x height, height being flipped)
    :param num_cameras: number of cameras
    :param cam_input_path: camera root directory path
    :param config_input_filename: config file name (found in respective camera folders in cam_input_path)
    :return: returns array of projected voxel image points for every camera
    """
    # Lookup entry for every camera
    lookup_table = []
    for camera in range(num_cameras):
        # Load camera parameters
        config_path = os.path.join(cam_input_path, "cam" + str(camera+1))
        mtx, dist, rvecs, tvecs = load_config_info(config_path, config_input_filename)

        # Project 3D voxel points to image plane and store them
        image_points_voxels, _ = cv2.projectPoints(voxel_points, rvecs, tvecs, mtx, dist)
        lookup_table.append(image_points_voxels)

    return np.array(lookup_table, dtype=np.float32)


def update_visible_voxels_and_extract_colors(voxel_volume_shape, lookup_table, fg_masks, images):
    """
    Updates visibility for voxels by checking if they are visible by all camera views as foreground and extracts colors
    for every camera view for turned on voxels.

    :param voxel_volume_shape: voxel volume shape dimensions (width, height, depth)
    :param lookup_table: lookup table of projected voxel image points per camera
    :param fg_masks: foreground masks for every camera
    :param images: images to get colors from for every camera
    :return: returns 3D array of booleans indicating whether a voxel is seen by all camera views as foreground or not
             (dimensions as width x depth x height) and 3D array of colors for all visible voxels for every camera view
             (dimensions as width x depth x height x camera)
    """
    # Volume shape dimensions
    width = voxel_volume_shape[0]
    height = voxel_volume_shape[1]
    depth = voxel_volume_shape[2]

    # Storing voxel visibility as True if voxel is turned on, False if not
    voxels_on = np.ones((width, depth, height), dtype=bool)
    # Storing colors for each voxel for each camera
    voxels_on_colors = np.empty((width, depth, height, len(fg_masks), 3), dtype=np.uint8)

    # Set voxels to off if they are not visible in every camera as foreground
    for camera in range(len(fg_masks)):
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxels_on[x, z, y]:
                        continue
                    # Get voxel index
                    voxel_idx = z + y * depth + x * (depth * height)
                    try:
                        # Voxel projection on image plane
                        x_im = int(lookup_table[camera][voxel_idx][0][0])
                        y_im = int(lookup_table[camera][voxel_idx][0][1])

                        # Check if voxel projection is within image boundaries and if it is visible from current camera
                        if not (0 <= y_im < fg_masks[camera].shape[0] and 0 <= x_im < fg_masks[camera].shape[1] and
                                fg_masks[camera][y_im, x_im] > 0):
                            # If a voxel is not visible for a camera then it is turned off and its color is removed
                            voxels_on[x, z, y] = False
                            voxels_on_colors[x, z, y, camera] = np.zeros((3,), dtype=np.uint8)
                        else:
                            # Store color from color frame for current camera
                            voxels_on_colors[x, z, y, camera] = images[camera][y_im, x_im, :].astype(np.uint8)
                    # Cases where points project to infinity, turn voxel off
                    except OverflowError:
                        voxels_on[x, z, y] = False
                        voxels_on_colors[x, z, y, camera] = np.zeros((3,), dtype=np.uint8)

    return voxels_on, voxels_on_colors


def plot_marching_cubes(voxels_on, rotate=True, plot_output_path="plots", plot_output_filename="marching_cubes.png"):
    """
    Runs marching cubes algorithm on activated voxels and plots results.

    :param voxels_on: 3D array of booleans indicating whether a voxel is seen by all camera views as foreground or not
                      (dimensions as width x depth x height)
    :param rotate: if True then rotates figure in plot to view it from the front, otherwise viewing it from the back
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    # Change orientation with rotation of 180 degrees
    if rotate:
        voxels_on = np.rot90(voxels_on, 2)

    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(voxels_on, 0)

    # Plot results from algorithm
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    # Set axes
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_xlim(voxels_on.shape[0], 0)
    ax.set_ylim(0, voxels_on.shape[1])
    ax.set_zlim(0, voxels_on.shape[2])

    # Adjust plot
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))
