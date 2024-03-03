import glm
import random
import numpy as np
import cv2
import os
import camera_calibration
import background_subtraction
import voxel_reconstruction
import utils

block_size = 1.0

# Parameters for voxel positions function
# Initialization with loading videos and training background models
initialized = False
videos = []
bg_models = []
# Background model parameters for every camera
# figure_threshold, figure_inner_threshold,
# apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
cam_bg_model_params = [
    [5000, 115, False, False, True, True],
    [5000, 115, False, False, True, True],
    [5000, 175, False, True, True, True],
    [5000, 115, False, False, False, True]
]
# Currently loaded frames and their index
current_frames = []
frame_count = 0
previous_masks = []
# Lookup table for voxels
lookup_table = None
voxel_points = None
voxel_size = 30


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
    global block_size, voxel_size, initialized, videos, bg_models, cam_bg_model_params, current_frames, frame_count, \
        lookup_table, voxel_points

    # Check whether initialization with loading videos and training background models has already been done
    if not initialized:
        # Load videos and train background models for every camera
        for camera in range(4):
            directory = os.path.join("data", "cam" + str(camera+1))

            # Load video
            videos.append(cv2.VideoCapture(os.path.join(directory, "video.avi")))

            # Give frame count of video as history for background model training
            _, _, frame_count = utils.get_video_properties(directory, "background.avi")
            # Train background model
            bg_models.append(background_subtraction.train_MOG_background_model(directory, "background.avi",
                                                                               use_hsv=True, history=frame_count,
                                                                               n_mixtures=50, bg_ratio=0.90,
                                                                               noise_sigma=0))

        # Calculate voxel volume
        voxel_points = voxel_reconstruction.create_voxel_volume((width, height, depth), voxel_size, block_size)

        # Create lookup table
        lookup_table = voxel_reconstruction.create_lookup_table(voxel_points, 4, "data", "config.xml")

        # Flag initialization is complete
        initialized = True

    # Read next frame in video
    current_frames = [video.read()[1] for video in videos]
    if any(frame is None for frame in current_frames):
        return [], []
    frame_count += 1

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
                    colors.append([voxels_on_colors[x, z, y][1][::-1] / 255.0])
                    #colors.append([x / width, z / depth, y / height])

    return data, colors


def get_cam_positions():
    """
    Calculates positions of cameras with rotation and translation vectors. Swaps Y and Z axis to convert OpenCV
    3D coordinate system to OpenGL and makes the new Y negative to face the viewer.

    :return: returns position for every camera and color vector for every camera
    """
    # Fix scale back to 1 unit
    _, chessboard_square_size = camera_calibration.load_chessboard_info("data", "checkerboard.xml")
    scale = 1.0 / chessboard_square_size

    # Get all camera positions
    camera_positions = []
    for camera in range(4):
        # Get camera rotation and translation
        _, _, rvecs, tvecs = voxel_reconstruction.load_config_info(os.path.join("data", "cam" + str(camera+1)),
                                                                                "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Get camera position
        position = -np.matrix(rmtx).T * np.matrix(tvecs) * scale

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
