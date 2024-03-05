import glm
import numpy as np
import cv2
import os
import background_subtraction
import voxel_reconstruction
import utils

block_size = 1.0

# Parameters for voxel positions function
# Initialization with loading videos and training background models
initialized = False
videos = []
bg_models = []
"""
# Background model parameters for every camera to fit 1 figure in grid
# figure_threshold, figure_inner_threshold,
# apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
cam_bg_model_params = [
    [5000, 115, False, False, True, True],
    [5000, 115, False, False, True, True],
    [5000, 175, False, True, True, True],
    [5000, 115, False, False, False, True]
]
bg_models_choice = "mog"
"""
# Background model parameters for every camera to fit 4 figures in grid
# figure_threshold, figure_inner_threshold,
# apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
cam_bg_model_params = [
    [1500, 500, False, True, True, True],
    [1500, 500, False, True, True, True],
    [1500, 500, False, True, True, True],
    [1500, 500, False, True, True, True]
]
bg_models_choice = "mog2"
# Currently loaded frames and their index
frame_count = 0
frame_interval = 50
# Lookup table for voxels
lookup_table = None
voxel_points = None
# Voxel size to fit 1 figure in grid
# voxel_size = 30
# Voxel size to fit 4 figures in grid
voxel_size = 45


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    """
    Sets voxels to be viewed and their colors. Voxels must be visible in all 4 foreground masks extracted by background
    subtraction models from video frames. Each time this function is called, the video moves to the next frame.

    :param width: determines voxel volume width
    :param height: determines voxel volume height
    :param depth: determines voxel volume depth
    :return: returns visible voxel data and colors
    """
    global initialized, lookup_table, voxel_points, voxel_size, block_size, \
        bg_models, bg_models_choice, cam_bg_model_params, videos, frame_count, frame_interval

    # Check whether initialization with loading videos and training background models has already been done
    if not initialized:
        print("Initializing background models, voxel volume, and lookup table.")
        # Load videos and train background models for every camera
        for camera in range(4):
            directory = os.path.join("data", "cam" + str(camera+1))

            # Load video
            videos.append(cv2.VideoCapture(os.path.join(directory, "video.avi")))

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

        # Create lookup table
        lookup_table = voxel_reconstruction.create_lookup_table(voxel_points, 4, "data", "config.xml")

        # Flag initialization is complete
        initialized = True

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
                return [], []

            # Check if frame will be used according to interval
            if frame_count_video % frame_interval == 0:
                current_frames.append(current_frame)
                break

            frame_count_video += 1
    if frame_count == 0:
        frame_count += 1
    else:
        frame_count += frame_interval

    # Extract foreground mask from video frame for each camera
    current_fg_masks = []
    for camera, current_frame in enumerate(current_frames):
        # figure_threshold, figure_inner_threshold,
        # apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
        params = cam_bg_model_params[camera]

        # Extract foreground mask
        current_fg_masks.append(np.array(
            background_subtraction.extract_foreground_mask(current_frame, bg_models[camera], 0, params[0], params[1],
                                                           params[2], params[3], params[4], params[5])))

    # Get voxels that are on and their colors from each camera
    print("\nFinding visible voxels for frame " + str(frame_count-1) +
          " (second " + str(int((frame_count-1)/frame_interval)) + ") of videos.")
    voxels_on, voxels_on_colors\
        = voxel_reconstruction.update_visible_voxels_and_extract_colors((width, height, depth), lookup_table,
                                                                        current_fg_masks, current_frames)

    # Format voxels that are on for viewing
    data = []
    colors = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxels_on[x, z, y]:
                    data.append([x * block_size - width / 2,
                                 y * block_size,
                                 z * block_size - depth / 2])

                    # Use color of only 2nd camera (front) and convert to 0-1
                    #colors.append([voxels_on_colors[x, z, y][1][::-1] / 255.0])
                    # Use generic coloring
                    colors.append([x / width, z / depth, y / height])

    # Plot marching cubes algorithm results
    if frame_count-1 == 0:
        print("Running marching cubes algorithm for frame " + str(frame_count-1) +
              " (second " + str(int((frame_count-1)/frame_interval)) + ") of videos.")
        voxel_reconstruction.plot_marching_cubes(voxels_on, rotate=True, plot_output_filename="marching_cubes_front")
        voxel_reconstruction.plot_marching_cubes(voxels_on, rotate=False, plot_output_filename="marching_cubes_back")

    return data, colors


def get_cam_positions():
    """
    Calculates positions of cameras with rotation and translation vectors. Swaps Y and Z axis to convert OpenCV
    3D coordinate system to OpenGL and makes the new Y negative to face the viewer.

    :return: returns position for every camera and color vector for every camera
    """
    global voxel_size

    # Get all camera positions
    camera_positions = []
    for camera in range(4):
        # Get camera rotation and translation
        _, _, rvecs, tvecs = voxel_reconstruction.load_config_info(os.path.join("data", "cam" + str(camera+1)),
                                                                   "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Get camera position
        position = -np.matrix(rmtx).T * np.matrix(tvecs/voxel_size)

        # Swap Y and Z axis for OpenGL system and make new Y negative to face the viewer
        camera_positions.append([position[0][0], -position[2][0], position[1][0]])

    return camera_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    """
    Calculates rotations of cameras with rotation vectors. Swaps Y and Z axis to convert OpenCV 3D coordinate system to
    OpenGL and makes the new Y negative to face the viewer.

    :return: returns rotation for every camera
    """
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
    for camera in range(4):
        # Get camera rotation
        _, _, rvecs, _ = voxel_reconstruction.load_config_info(os.path.join("data", "cam" + str(camera+1)),
                                                               "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Convert OpenCV rotation matrix (row-major) to OpenGL rotation matrix (column-major)
        # Swap Y and Z positions of rotation to account for conversion
        cam_rotation = axes_conversion * glm.mat4(rmtx[0][0], rmtx[1][0], rmtx[2][0], 0,
                                                  rmtx[0][2], rmtx[1][2], rmtx[2][2], 0,
                                                  rmtx[0][1], rmtx[1][1], rmtx[2][1], 0,
                                                  0, 0, 0, 1)
        cam_rotations.append(cam_rotation)

    return cam_rotations
