import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linear_sum_assignment
from scipy import interpolate
from skimage import measure
from copy import deepcopy
import utils


def load_config_info(config_input_path="data/cam", config_input_filename="config.xml"):
    """
    Loads intrinsic (camera matrix, distortion coefficients) and extrinsic (rotation vector, translation vector) camera
    parameters from config file.

    :param config_input_path: config xml file directory path
    :param config_input_filename: config xml file name
    :return: camera matrix
    """
    # Select tags for loaded nodes and their types
    node_tags = ["CameraMatrix", "DistortionCoeffs", "RotationVector", "TranslationVector"]
    node_types = ["mat" for _ in range(len(node_tags))]

    # Load nodes
    nodes = utils.load_xml_nodes(config_input_path, config_input_filename, node_tags, node_types)

    # Parse config
    mtx = nodes.get("CameraMatrix")
    dist = nodes.get("DistortionCoeffs")
    rvecs = nodes.get("RotationVector")
    tvecs = nodes.get("TranslationVector")

    return mtx, dist, rvecs, tvecs


def create_voxel_volume(voxel_volume_shape, voxel_size, block_size=1.0):
    """
    Creates voxel volume points given dimensions.

    :param voxel_volume_shape: voxel volume shape dimensions (width, height, depth)
    :param voxel_size: distance in mm for voxels
    :param block_size: grid block size unit
    :return: returns 3D voxel volume points (dimensions as width x depth x height, height being flipped)
    """
    # Volume shape dimensions
    width, height, depth = voxel_volume_shape

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


def create_lookup_table(voxel_points, cam_choice, cam_input_path="data", config_input_filename="config.xml"):
    """
    Creates lookup table of 3D voxels projected to 2D image points for a number of camera views.

    :param voxel_points: 3D voxel volume points (dimensions as width x depth x height, height being flipped)
    :param cam_choice: selected cameras (numbers correspond to directories)
    :param cam_input_path: camera root directory path
    :param config_input_filename: camera config file name (found in respective camera directories in cam_input_path)
    :return: returns array of projected voxel image points for every camera
    """
    # Lookup entry for every camera
    lookup_table = []
    for camera in cam_choice:
        # Load camera parameters
        config_path = os.path.join(cam_input_path, "cam" + str(camera))
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
    width, height, depth = voxel_volume_shape

    # Storing voxel visibility as True if voxel is turned on, False if not
    voxel_visibility = np.ones((width, depth, height), dtype=bool)
    # Storing colors for each voxel for each camera
    voxel_cam_colors = np.empty((width, depth, height, len(fg_masks), 3), dtype=np.uint8)

    # Set voxels to off if they are not visible in every camera as foreground
    for camera in range(len(fg_masks)):
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_visibility[x, z, y]:
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
                            voxel_visibility[x, z, y] = False
                            voxel_cam_colors[x, z, y, 0] = np.zeros((3,), dtype=np.uint8)
                        else:
                            # Store color from color frame for current camera
                            voxel_cam_colors[x, z, y, camera] = images[camera][y_im, x_im, :].astype(np.uint8)
                    # Cases where points project to infinity, turn voxel off
                    except OverflowError:
                        voxel_visibility[x, z, y] = False
                        voxel_cam_colors[x, z, y, camera] = np.zeros((3,), dtype=np.uint8)

    return voxel_visibility, voxel_cam_colors


def position_and_color_visible_voxels(voxel_volume_shape, voxel_visibility, voxel_cam_colors=None,
                                      voxel_cam_colors_idx=-1, voxel_cam_colors_unit=True, block_size=1.0):
    """
    Centers visible voxel positions around voxel volume center and colors each one. Coloring includes generic coloring
    of each axis divided by the corresponding voxel volume shape dimension (X / width, Z / depth, Y / height) and
    coloring from a camera view if selected by parameters.

    :param voxel_volume_shape: voxel volume shape dimensions (width, height, depth)
    :param voxel_visibility: 3D array of booleans indicating whether a voxel is seen by all camera views as foreground
                             or not (dimensions as width x depth x height)
    :param voxel_cam_colors: 3D array of colors for all visible voxels for every camera view (dimensions as width x
                             depth x height x camera) to get colors for a selected camera with voxel_cam_colors_idx or
                             None to ignore colors from camera views
    :param voxel_cam_colors_idx: camera view index to get color for each voxel from voxel_cam_colors, if -1 then
                                 colors from camera views are ignored
    :param voxel_cam_colors_unit: converts colors from camera view to 0-1 range if True, otherwise keeps 0-255 range
    :param block_size: grid block size unit
    :return: returns array of 3D visible voxel points, array of 3D generic colors for each visible voxel, array of
             3D colors for each visible voxel from selected camera or None if parameters ignore colors from camera views
    """
    # Volume shape dimensions
    width, height, depth = voxel_volume_shape

    # Center voxels around center of volume and choose colors
    visible_voxel_points = []
    visible_voxel_colors = []
    if voxel_cam_colors_idx != -1 and voxel_cam_colors is not None:
        visible_voxel_colors_cam = []
    else:
        visible_voxel_colors_cam = None
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_visibility[x, z, y]:
                    visible_voxel_points.append([x * block_size - width / 2,
                                                 y * block_size,
                                                 z * block_size - depth / 2])

                    # Get generic coloring
                    visible_voxel_colors.append([x / width, z / depth, y / height])

                    # Get color from selected camera view
                    # Convert color to 0-1 range and reverse order
                    if voxel_cam_colors_unit:
                        visible_voxel_colors_cam.append((voxel_cam_colors[x, z, y][voxel_cam_colors_idx][::-1] / 255.0)
                                                        .tolist())
                    # Keep color in 0-255 range and reverse order
                    else:
                        visible_voxel_colors_cam.append((voxel_cam_colors[x, z, y][voxel_cam_colors_idx][::-1])
                                                        .tolist())

    return visible_voxel_points, visible_voxel_colors, visible_voxel_colors_cam


def plot_marching_cubes_surface_mesh(voxel_visibility, rotate=True, plot_output_path="plots",
                                     plot_output_filename="marching_cubes.png"):
    """
    Runs marching cubes algorithm on activated voxels and plots extracted surface mesh.

    :param voxel_visibility: 3D array of booleans indicating whether a voxel is seen by all camera views as foreground
                             or not (dimensions as width x depth x height)
    :param rotate: if True then rotates figure in plot to view it from the front, otherwise viewing it from the back
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    # Change orientation with rotation of 180 degrees
    if rotate:
        voxel_visibility = np.rot90(voxel_visibility, 2)

    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(voxel_visibility, 0)
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")

    # Plot surface mesh
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(mesh)

    # Set axes
    ax.set_title("Surface Mesh from Marching Cubes")
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Y (height)")
    ax.set_xlim(voxel_visibility.shape[0], 0)
    ax.set_ylim(0, voxel_visibility.shape[1])
    ax.set_zlim(0, voxel_visibility.shape[2])

    # Adjust plot
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()


def cluster_visible_voxels(visible_voxel_points, visible_voxel_points_colors=None, cluster_num=4, attempts=10,
                           iterations=100, outlier_std_away=2.0):
    """
    Clusters voxel positions using K-means clustering. The height is ignored and only width and depth are taken into
    account.

    :param visible_voxel_points: array of 3D visible voxel positions
    :param visible_voxel_points_colors: array of arrays of 3D colors for each visible voxel (multiple colorings), if
                                        None then coloring is ignored
    :param cluster_num: number of clusters
    :param attempts: number of times kmeans algorithm is executed using different initial labellings to avoid local
                     minima
    :param iterations: number of iterations for each kmeans algorithm attempt
    :param outlier_std_away: if larger than 0 then outlier removal is performed after clustering voxel points, where
                             points at this number of the standard deviations of the mean distance from their relative
                             cluster center or farther away are removed and the points that are left are then
                             re-clustered
    :return: returns array of 2D cluster centers, array of arrays of clusters and their 3D voxel points, array of arrays
             of arrays of clusters and their 3D colors (colorings x clusters x colors) or None if coloring is ignored,
             array of outlier 3D voxel points (empty if not handling outliers)
    """
    # Ignore the height, keep width and depth
    voxel_points = np.array(visible_voxel_points, dtype=np.float32)
    if voxel_points.shape[1] == 3:
        voxel_points_xz = voxel_points[:, [0, 2]].astype(np.float32)
    else:
        voxel_points_xz = voxel_points

    # Handle colorings
    if visible_voxel_points_colors is not None:
        voxel_points_colors = np.array([np.array(coloring, dtype=np.float32) for coloring in visible_voxel_points_colors],
                                       dtype=np.float32)
    else:
        voxel_points_colors = None

    # Run K-means clustering
    _, labels, centers = cv2.kmeans(voxel_points_xz, cluster_num, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 0.2),
                                    attempts, cv2.KMEANS_PP_CENTERS)

    # Split to separate cluster lists
    clusters = [[] for _ in range(cluster_num)]
    if visible_voxel_points_colors is not None:
        clusters_colors = [[[] for _ in range(cluster_num)] for _ in range(len(voxel_points_colors))]
    else:
        clusters_colors = None
    for voxel_position_idx, label in enumerate(labels):
        clusters[label[0]].append(voxel_points[voxel_position_idx])

        if visible_voxel_points_colors is not None:
            for coloring_idx, coloring in enumerate(voxel_points_colors):
                clusters_colors[coloring_idx][label[0]].append(coloring[voxel_position_idx])
    clusters = [np.array(cluster_voxels) for cluster_voxels in clusters]
    if visible_voxel_points_colors is not None:
        clusters_colors = [[np.array(cluster_voxels_colors) for cluster_voxels_colors in coloring]
                           for coloring in clusters_colors]

    # Remove outliers from each cluster
    voxel_points_clean = []
    voxel_points_outliers = []
    if visible_voxel_points_colors is not None:
        voxel_points_clean_colors = [[] for _ in range(len(clusters_colors))]
    else:
        voxel_points_clean_colors = None
    if outlier_std_away > 0:
        for cluster_idx, cluster_voxels in enumerate(clusters):
            # Find distances of voxel positions from cluster center
            distances_to_center = np.linalg.norm(cluster_voxels[:, [0, 2]] - centers[cluster_idx], axis=1)

            # Calculate mean and standard deviation for distances
            mean_distance = np.mean(distances_to_center)
            std_distance = np.std(distances_to_center)

            # Mark outliers as positions at a number of the standard deviations of the mean distance or farther away
            outliers = distances_to_center >= mean_distance + std_distance * outlier_std_away

            # Remove outliers from the cluster and save positions that were and weren't outliers
            for voxel_idx, (voxel, outlier) in enumerate(zip(cluster_voxels, outliers)):
                if not outlier:
                    voxel_points_clean.append(voxel)
                    if visible_voxel_points_colors is not None:
                        for coloring_idx, coloring in enumerate(clusters_colors):
                            voxel_points_clean_colors[coloring_idx].append(coloring[cluster_idx][voxel_idx])
                else:
                    voxel_points_outliers.append(voxel)

        # Re-cluster the voxel positions that weren't outliers
        centers, clusters, clusters_colors, _ = cluster_visible_voxels(voxel_points_clean, voxel_points_clean_colors,
                                                                       outlier_std_away=0.0)

    return centers, clusters, clusters_colors, np.array(voxel_points_outliers, dtype=np.float32)


def plot_visible_voxel_clusters(centers, clusters, outliers=None, labels=None, color_palette=None,
                                plot_output_path="plots", plot_output_filename="voxel_clusters.png"):
    """
    Plots results of clustering voxel positions.

    :param clusters: array of arrays of clusters and their 3D voxel points
    :param centers: array of 2D cluster centers
    :param outliers: array of outlier 3D voxel points
    :param labels: labels for coloring clusters according to color palette indexes, if None then coloring in order
    :param color_palette: color palette to choose cluster color using modulo, if None then uses a default palette
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    # Default palette
    if color_palette is None:
        color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                         [1, 0.5, 0], [0, 0.5, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]

    # Create plot figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Plot clusters
    for cluster_idx, cluster_voxels in enumerate(clusters):
        # Color in order of index
        if labels is None:
            color = color_palette[cluster_idx % len(color_palette)]
        # Match label to color
        else:
            color = color_palette[labels[cluster_idx] % len(color_palette)]

        ax.scatter(cluster_voxels[:, 0], cluster_voxels[:, 2],
                   color=color, marker="o", label="Cluster " + str(cluster_idx))

    # Plot cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], s=100, color="black", marker="x", label="Centers")

    # Plot outliers
    if outliers is not None:
        ax.scatter(outliers[:, 0], outliers[:, 2], color="darkgray", marker="o", label="Outliers")

    # Set axes
    ax.set_title("Clustered Voxel Points")
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Z (depth)")
    ax.invert_yaxis()
    ax.legend()

    # Adjust plot
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()


def create_color_models(clusters, voxel_size, images, cam_choice, cam_input_path="data",
                        config_input_filename="config.xml", crop_height_top=0.2, crop_height_bottom=0.45,
                        crop_width_left=0.2, crop_width_right=0.2, output_projections=False,
                        output_projections_filename="color_model_projections.jpg"):
    """
    Creates color models from 3D voxels of clusters projected to 2D image points for a number of camera views. Each
    color model consists of HSV color histograms.

    :param clusters: array of arrays of clusters and their voxel points
    :param voxel_size: distance in mm for voxels
    :param images: images to get colors from for every camera
    :param cam_choice: selected cameras (numbers correspond to directories)
    :param cam_input_path: camera root directory path
    :param config_input_filename: camera config file name (found in respective camera directories in cam_input_path)
    :param crop_height_top: array of fractions of cluster projected image points to crop from the top for every camera,
                            if None then set to 0 for every camera and no cropping is applied
    :param crop_height_bottom: array of fractions of cluster projected image points to crop from the bottom for every
                               camera, if None then set to 0 for every camera and no cropping is applied
    :param crop_width_left: array of fractions of cluster projected image points to crop from the left for every camera,
                            if None then set to 0 for every camera and no cropping is applied
    :param crop_width_right: array of fractions of cluster projected image points to crop from the right for every
                             camera, if None then set to 0 for every camera and no cropping is applied
    :param output_projections: if True then outputs image with selected image points of voxel projections for every
                               camera in respective camera directories in cam_input_path
    :param output_projections_filename: output image file name (including extension), used when output_projections is
                                        True
    :return: returns array of arrays of color model histograms for every camera
    """
    if crop_height_top is None:
        crop_height_top = [0 for _ in range(len(cam_choice))]
    if crop_height_bottom is None:
        crop_height_bottom = [0 for _ in range(len(cam_choice))]
    if crop_width_left is None:
        crop_width_left = [0 for _ in range(len(cam_choice))]
    if crop_width_right is None:
        crop_width_right = [0 for _ in range(len(cam_choice))]

    # Create color models for each camera view
    color_models = [[] for _ in range(len(cam_choice))]
    for camera_idx, camera in enumerate(cam_choice):
        # Convert image to HSV
        image_hsv = cv2.cvtColor(images[camera_idx], cv2.COLOR_BGR2HSV)

        # Image with selected image points of voxel projections
        if output_projections:
            image_projections = deepcopy(images[camera_idx])
        else:
            image_projections = None

        # Create color model for each cluster
        for cluster_voxels in clusters:
            # Change dimensions and scale
            cluster_voxels_reformat = np.array([[voxel[0], voxel[2], -voxel[1]] for voxel in cluster_voxels])*voxel_size

            # Map 3D voxels in cluster to 2D image points for camera view
            cluster_image_points = create_lookup_table(cluster_voxels_reformat, [camera], cam_input_path,
                                                       config_input_filename)[0]

            # Find cluster's min and max height and width, origin at top-left
            y_coordinates = cluster_image_points[:, 0, 1].astype(int)
            x_coordinates = cluster_image_points[:, 0, 0].astype(int)
            # Origin at top-left so minimum height is max value and maximum height is min value
            min_height = np.max(y_coordinates)
            max_height = np.min(y_coordinates)
            min_width = np.min(x_coordinates)
            max_width = np.max(x_coordinates)

            # Crop voxel space
            total_height = (max_height - min_height + 1)
            min_height += total_height * crop_height_bottom[camera_idx]
            max_height -= total_height * crop_height_top[camera_idx]
            total_width = (max_width - min_width + 1)
            min_width += total_width * crop_width_left[camera_idx]
            max_width -= total_width * crop_width_right[camera_idx]

            # Keep image points within cropped ranges, avoid duplicates
            color_model_points = set()
            for image_points in cluster_image_points:
                x_im = int(image_points[0][0])
                y_im = int(image_points[0][1])
                # Check if image point is within image boundaries
                im_are_within_image = 0 <= y_im < image_hsv.shape[0] and 0 <= x_im < image_hsv.shape[1]
                # Check if image point is within crop range
                # Origin at top-left so minimum height is larger than maximum height
                im_are_within_crop = max_height <= y_im <= min_height and min_width <= x_im <= max_width
                if im_are_within_image and im_are_within_crop:
                    color_model_points.add((x_im, y_im))
                    if output_projections:
                        # Plot the image points on the image
                        cv2.circle(image_projections, (x_im, y_im), 2, (0, 255, 0), -1)
            color_model_points = np.array(list(color_model_points))

            # No points passed checks
            if color_model_points.size == 0:
                color_models[camera_idx].append(None)
                continue

            # Extract HSV values from image points
            h, s, v = image_hsv[color_model_points[:, 1], color_model_points[:, 0]].T

            # Create histogram of 180 x 256 x 256 = 11,796,480 colors
            # Using 180 bins for H to fit into single byte (half of color wheel)
            hist_h = cv2.calcHist([h.astype(np.float32)], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([s.astype(np.float32)], [0], None, [256], [0, 256])
            hist_v = cv2.calcHist([v.astype(np.float32)], [0], None, [256], [0, 256])

            # Normalize between 0 and 1
            hist_h = cv2.normalize(hist_h, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            hist_s = cv2.normalize(hist_s, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            hist_v = cv2.normalize(hist_v, None, 0, 1, norm_type=cv2.NORM_MINMAX)

            # Store color model
            color_models[camera_idx].append([hist_h, hist_s, hist_v])

        # Output image points of color models for camera view
        if output_projections:
            cam_path = os.path.join(cam_input_path, "cam" + str(camera))
            cv2.imwrite(os.path.join(cam_path, output_projections_filename), image_projections)

    return color_models


def match_color_models(online_color_models, offline_color_models):
    """
    Matches online color models to offline color models. Matching is calculated using the Hungarian algorithm and a
    distance matrix between every online model and offline model (online x offline). The distance matrix is calculated
    by aggregating Bhattacharyya distances across histogram channels and camera views.

    :param online_color_models: array of arrays of online color model histograms for every camera
    :param offline_color_models: array of arrays of offline color model histograms for every camera
    :return: returns indexes of offline color models that online color models matched to
    """
    # Get sizes
    cam_num = len(offline_color_models)
    online_num = len(online_color_models[0])
    offline_num = len(offline_color_models[0])
    channel_num = len(online_color_models[0][0])

    # Create distance matrix
    distance_mtx = np.zeros((online_num, offline_num))

    # Aggregate histograms across cameras and channels for every position
    for online_idx in range(online_num):
        for offline_idx in range(offline_num):
            hist_distance = 0
            for camera_idx in range(cam_num):
                hist_online = online_color_models[camera_idx][online_idx]
                hist_offline = offline_color_models[camera_idx][offline_idx]
                for channel_idx in range(channel_num):
                    hist_distance += cv2.compareHist(hist_online[channel_idx], hist_offline[channel_idx],
                                                     method=cv2.HISTCMP_BHATTACHARYYA)
            distance_mtx[online_idx, offline_idx] = hist_distance

    # Apply the Hungarian algorithm to match online models to offline models
    _, offline_idx_matches = linear_sum_assignment(distance_mtx)

    return offline_idx_matches


def label_visible_voxels_with_color(clusters, labels, color_palette=None):
    """
    Colors visible voxel positions in clusters according to labels for the clusters.

    :param clusters: array of arrays of clusters and their 3D voxel points
    :param labels: labels for coloring clusters according to color palette indexes, if None then coloring in order
    :param color_palette: color palette to choose cluster color using modulo, if None then uses a default palette
    :return: returns array of 3D voxel points of every cluster, array of 3D label colors for each voxel
    """
    # Default palette
    if color_palette is None:
        color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                         [1, 0.5, 0], [0, 0.5, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]

    # Match labels to colors
    label_colors = [color_palette[label % len(color_palette)] for label in labels]

    # Colors each voxel according to label
    clusters_voxel_points = []
    clusters_voxel_points_label_colors = []
    for cluster_idx, cluster_voxels in enumerate(clusters):
        for voxel in cluster_voxels:
            clusters_voxel_points.append(voxel)
            clusters_voxel_points_label_colors.append(label_colors[cluster_idx])

    return clusters_voxel_points, clusters_voxel_points_label_colors


def update_cluster_center_trajectories(centers, trajectories, labels):
    """
    Updates trajectories based on cluster centers.

    :param centers: array of 2D cluster centers
    :param trajectories: array of arrays of 2D trajectory points for every cluster
    :param labels: labels for coloring clusters according to color palette indexes, if None then coloring in order
    :return:
    """
    for cluster_idx, center in enumerate(centers):
        # Color in order of index
        if labels is None:
            trajectories[cluster_idx].append(center)
        # Match label to color
        else:
            trajectories[labels[cluster_idx]].append(center)

    return trajectories


def plot_cluster_center_trajectories(trajectories, labels=None, color_palette=None, interpolation_samples=0,
                                     plot_output_path="plots", plot_output_filename="trajectories.png"):
    """
    Plots every trajectory point to visualize paths. Interpolation can be used to fill in empty spaces.

    :param trajectories: array of arrays of 2D trajectory points for every cluster
    :param labels: labels for coloring clusters according to color palette indexes, if None then coloring in order
    :param color_palette: color palette to choose cluster color using modulo, if None then uses a default palette
    :param interpolation_samples: number of samples used for interpolation, if 0 then only raw trajectory points are
                                  used
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    :return:
    """
    # Default palette
    if color_palette is None:
        color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                         [1, 0.5, 0], [0, 0.5, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]

    # Create plot figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Plot clusters
    for cluster_idx in range(len(trajectories)):
        # Color in order of index
        if labels is None:
            color = color_palette[cluster_idx % len(color_palette)]

            # Apply interpolation
            if interpolation_samples > 0:
                tck, u = interpolate.splprep(np.array(trajectories[cluster_idx]).T, s=0)
                x_points, y_points = interpolate.splev(np.linspace(0, 1, interpolation_samples), tck)
            # Keep raw points
            else:
                x_points = [position[0] for position in trajectories[cluster_idx]]
                y_points = [position[1] for position in trajectories[cluster_idx]]
        # Match label to color
        else:
            color = color_palette[labels[cluster_idx] % len(color_palette)]

            # Apply interpolation
            if interpolation_samples > 0:
                tck, u = interpolate.splprep(np.array(trajectories[labels[cluster_idx]]).T, s=0)
                x_points, y_points = interpolate.splev(np.linspace(0, 1, interpolation_samples), tck)
            # Keep raw points
            else:
                x_points = [position[0] for position in trajectories[labels[cluster_idx]]]
                y_points = [position[1] for position in trajectories[labels[cluster_idx]]]
        ax.scatter(x_points, y_points, color=color, marker="s", s=10)

    # Set axes
    ax.set_title("Trajectory Points for Every Figure")
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Z (depth)")
    ax.invert_yaxis()
    # Set the aspect ratio to be equal
    ax.set_aspect("equal")

    # Adjust plot
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()













