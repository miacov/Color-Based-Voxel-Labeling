import glm
import random
import numpy as np
import cv2
import os
import background_subtraction
import voxel_reconstruction
import utils

# Seed for reproducibility
random.seed(0)

# Grid block size unit
block_size = 1.0

# Cameras used for voxel reconstruction
cam_reconstruction = [1, 2, 3, 4]
# Camera used to color voxels if not making color models
cam_color = 2
# Color palette for visualization
color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                 [1, 0.5, 0], [0, 0.5, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]

# Handle parameters for figures in view
figure_num = 4
if figure_num == 1:
    # Voxel size to fit 1 figure in grid
    voxel_size = 30

    # Background model parameters for every camera to fit 1 figure in grid
    # figure_threshold, figure_inner_threshold,
    # apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post, apply_dilation_post
    cam_bg_model_params = [
        [5000, 115, False, False, True, True, False],
        [5000, 115, False, False, True, True, False],
        [5000, 175, False, True, True, True, False],
        [5000, 115, False, False, False, True, False]
    ]
    bg_models_choice = "mog"
else:
    # Voxel size to fit more than 1 figure in grid
    voxel_size = 60

    # Background model parameters for every camera to fit more than 1 figure in grid
    # figure_threshold, figure_inner_threshold,
    # apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post, apply_dilation_post
    cam_bg_model_params = [
        [1500, 500, False, True, True, True, False],
        [1500, 500, False, True, True, True, False],
        [1500, 500, False, True, True, True, False],
        [1500, 500, False, True, True, True, False]
    ]
    bg_models_choice = "mog2"

# Lookup table for voxels
lookup_table = None
voxel_points = None

# Videos to get frames
videos = []
fps = 0
# Currently loaded frames and their index
frame_count = 0
frame_interval = 50

# Background models
bg_models = []

# Color models
color_models = None

# Cameras used for color models
cam_color_models = [1, 2, 4]
# Color model cropping (top, bottom, left, right) for every camera
cam_color_model_crops = [
    [0.2, 0.2, 0.2],
    [0.45, 0.45, 0.45],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]


def generate_grid(width, depth):
    """
    Generates the floor grid locations.

    :param width: determines floor width
    :param depth: determines floor depth
    :return: returns array of 3D floor grid points and array of 3D colors for each grid position
    """
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    """
    Sets voxels to be viewed. Voxels must be visible in all foreground masks extracted by background subtraction models
    from video frames. Each time this function is called, the source video that determines visibility moves to the next
    frame. Clustering is performed to remove outlier voxels.

    :param width: determines voxel volume width
    :param height: determines voxel volume height
    :param depth: determines voxel volume depth
    :return: returns array of 3D visible voxel points and array of 3D colors for each visible voxel
    """
    global cam_reconstruction, cam_color, color_palette, figure_num, \
        lookup_table, voxel_points, voxel_size, block_size, videos, fps, frame_count, frame_interval, \
        bg_models, bg_models_choice, cam_bg_model_params, color_models, cam_color_models, cam_color_model_crops

    # Initialize background models, voxel volume, and lookup table
    if lookup_table is None:
        print("Initializing background models, voxel volume, and lookup table.")
        # Load videos and train background models for every camera
        for camera in cam_reconstruction:
            directory = os.path.join("data", "cam" + str(camera))

            # Load video
            videos.append(cv2.VideoCapture(os.path.join(directory, "video.avi")))
            if fps == 0:
                fps = videos[0].get(cv2.CAP_PROP_FPS)

            # Give frame count of video as history for background model training
            _, _, bg_frame_count = utils.get_video_properties(directory, "background.avi")
            # Train background model
            if bg_models_choice == "mog":
                bg_models.append(background_subtraction.train_MOG_background_model(directory, "background.avi",
                                                                                   use_hsv=True, history=bg_frame_count,
                                                                                   n_mixtures=50, bg_ratio=0.90,
                                                                                   noise_sigma=0))
            elif bg_models_choice == "mog2":
                bg_models.append(background_subtraction.train_MOG2_background_model(directory, "background.avi",
                                                                                    use_hsv=True,
                                                                                    history=bg_frame_count,
                                                                                    var_threshold=500,
                                                                                    detect_shadows=True))
            else:
                bg_models.append(background_subtraction.train_KNN_background_model(directory, "background.avi",
                                                                                   use_hsv=True, history=frame_count,
                                                                                   dist_threshold=3500,
                                                                                   detect_shadows=True))

        # Calculate voxel volume
        voxel_points = voxel_reconstruction.create_voxel_volume((width, height, depth), voxel_size, block_size)

        # Create lookup table to map 3D voxels to 2D image points for each camera view
        lookup_table = voxel_reconstruction.create_lookup_table(voxel_points, cam_reconstruction, "data", "config.xml")

    # Read next frame in every video with interval
    current_frames = []
    for video in videos:
        frame_count_video = frame_count
        while True:
            ret_frame, current_frame = video.read()

            # If any video ended then stop and show no voxels
            if not ret_frame:
                for cap in videos:
                    cap.release()

                # Reset everything to initial states to restart on next function call
                lookup_table = None
                voxel_points = None
                videos = []
                frame_count = 0
                bg_models = []
                color_models = None

                return [], [], [], [], [], [], [], [], True

            # Check if frame will be used according to interval
            if frame_count_video % frame_interval == 0:
                current_frames.append(current_frame)
                break

            frame_count_video += 1
    if frame_count == 0:
        is_first_frame = True
        frame_count += 1
    else:
        is_first_frame = False
        frame_count += frame_interval

    # Extract foreground mask from video frame for each camera
    current_fg_masks = []
    for camera_idx, current_frame in enumerate(current_frames):
        # figure_threshold, figure_inner_threshold,
        # apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
        bg_model_params = cam_bg_model_params[camera_idx]

        # Extract foreground mask
        current_fg_masks.append(background_subtraction.extract_foreground_mask(current_frame, bg_models[camera_idx], 0,
                                                                               *bg_model_params))

    # Get voxels that are on and their colors from each camera
    print("\nFinding visible voxels for frame " + str(frame_count-1) +
          " (second " + str(int((frame_count-1)/fps)) + ") of videos.")
    voxel_visibility, voxel_cam_colors \
        = voxel_reconstruction.update_visible_voxels_and_extract_colors((width, height, depth), lookup_table,
                                                                        current_fg_masks, current_frames)

    # Center activated voxel positions around voxel volume center and color using selected camera view
    visible_voxel_points, visible_voxel_colors, visible_voxel_colors_cam \
        = voxel_reconstruction.position_and_color_visible_voxels((width, height, depth), voxel_visibility,
                                                                 voxel_cam_colors, cam_color-1, True, block_size)

    # Cluster voxel voxel positions and plot results
    print("Clustering voxel positions for frame " + str(frame_count-1) +
          " (second " + str(int((frame_count-1)/fps)) + ") of videos.")
    centers, clusters, clusters_color, outliers, \
        = voxel_reconstruction.cluster_visible_voxels(visible_voxel_points,
                                                      [visible_voxel_colors, visible_voxel_colors_cam],
                                                      cluster_num=figure_num, outlier_std_away=1.5)

    # Create color models
    print("Creating and labeling color models for frame " + str(frame_count-1) +
          " (second " + str(int((frame_count-1)/fps)) + ") of videos.")
    frame_idx = [idx for idx, cam in enumerate(cam_reconstruction) if cam in cam_color_models]
    current_frames_color_models = [current_frames[idx] for idx in frame_idx]
    current_color_models = \
        voxel_reconstruction.create_color_models(clusters, voxel_size, current_frames_color_models, cam_color_models,
                                                 "data", "config.xml", *cam_color_model_crops,
                                                 output_projections=is_first_frame,
                                                 output_projections_filename="video_color_model_projections.jpg")

    # Store offline color models created using the first frame from each camera
    if color_models is None:
        color_models = current_color_models

        # Plot results of clustering for first frame
        voxel_reconstruction.plot_visible_voxel_clusters(centers, clusters, outliers, None, color_palette,
                                                         plot_output_filename="voxel_clusters_offline.png")

    # Match online color models to offline color models
    matches = voxel_reconstruction.match_color_models(current_color_models, color_models)

    # Label online color models with matches to offline color models
    clusters_visible_voxel_points, clusters_voxel_points_label_colors \
        = voxel_reconstruction.label_visible_voxels_with_color(clusters, matches, color_palette)

    # Flatten rest of color arrays
    clusters_visible_voxel_colors = np.concatenate(clusters_color[0])
    clusters_visible_voxel_colors_cam = np.concatenate(clusters_color[1])

    # Plot results of clustering with matched indexes for a certain frame to showcase outlier removal
    if frame_count-1 == 150:
        voxel_reconstruction.plot_visible_voxel_clusters(centers, clusters, outliers, matches, color_palette,
                                                         plot_output_filename="voxel_clusters_online.png")

    if is_first_frame:
        # Plot marching cubes algorithm results
        print("Running marching cubes algorithm for frame " + str(frame_count-1) +
              " (second " + str(int((frame_count-1)/fps)) + ") of videos to plot surface mesh.")
        voxel_reconstruction.plot_marching_cubes_surface_mesh(voxel_visibility, rotate=True,
                                                              plot_output_filename="marching_cubes_front")
        voxel_reconstruction.plot_marching_cubes_surface_mesh(voxel_visibility, rotate=False,
                                                              plot_output_filename="marching_cubes_back")

    return clusters_visible_voxel_points, clusters_visible_voxel_colors, clusters_visible_voxel_colors_cam,\
        clusters_voxel_points_label_colors, visible_voxel_points, visible_voxel_colors, visible_voxel_colors_cam, False


def get_cam_positions():
    """
    Calculates positions of cameras with rotation and translation vectors. Swaps Y and Z axis to convert OpenCV
    3D coordinate system to OpenGL and makes the new Y negative to face the viewer.

    :return: returns position for every camera and color vector for every camera
    """
    global cam_reconstruction, color_palette, voxel_size

    # Get all camera positions
    cam_positions = []
    cam_colors = []
    for camera in cam_reconstruction:
        # Get camera rotation and translation
        _, _, rvecs, tvecs = voxel_reconstruction.load_config_info(os.path.join("data", "cam" + str(camera)),
                                                                   "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Get camera position
        position = -np.matrix(rmtx).T * np.matrix(tvecs/voxel_size)

        # Swap Y and Z axis for OpenGL system and make new Y negative to face the viewer
        cam_positions.append([position[0][0], -position[2][0], position[1][0]])
        # Use color according to palette
        cam_colors.append(color_palette[(camera-1) % len(color_palette)])

    return cam_positions, cam_colors


def get_cam_rotation_matrices():
    """
    Calculates rotations of cameras with rotation vectors. Swaps Y and Z axis to convert OpenCV 3D coordinate system to
    OpenGL and makes the new Y negative to face the viewer.

    :return: returns rotation for every camera
    """
    global cam_reconstruction

    # Swap Y and Z axis for OpenGL system and make new Y negative to face the camera
    # Rotation matrix for rotating 90 degrees around Y to swap Y and Z
    rotate_90_y = glm.rotate(np.pi / 2.0, glm.vec3(0, 1, 0))
    # Flip new Y sign
    flip_y = glm.mat4(1, 0, 0, 0,
                      0, -1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1)
    # Combine the rotation and flip matrices
    axes_conversion = rotate_90_y * flip_y

    # Get all camera rotations
    cam_rotations = []
    for camera in cam_reconstruction:
        # Get camera rotation
        _, _, rvecs, _ = voxel_reconstruction.load_config_info(os.path.join("data", "cam" + str(camera)), "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Convert OpenCV rotation matrix (row-major) to OpenGL rotation matrix (column-major)
        # Swap Y and Z positions of rotation to account for conversion
        cam_rotation = axes_conversion * glm.mat4(rmtx[0][0], rmtx[1][0], rmtx[2][0], 0,
                                                  rmtx[0][2], rmtx[1][2], rmtx[2][2], 0,
                                                  rmtx[0][1], rmtx[1][1], rmtx[2][1], 0,
                                                  0, 0, 0, 1)
        cam_rotations.append(cam_rotation)

    return cam_rotations
