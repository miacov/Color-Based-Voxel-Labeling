import glm
import glfw
from engine.base.program import get_linked_program
from engine.renderable.model import Model
from engine.buffer.texture import *
from engine.buffer.hdrbuffer import HDRBuffer
from engine.buffer.blurbuffer import BlurBuffer
from engine.effect.bloom import Bloom
from assignment import set_voxel_positions, generate_grid, get_cam_positions, get_cam_rotation_matrices
from engine.camera import Camera
from engine.config import config
import OpenGL.GL as gl
from PIL import Image
import os
import cv2
import re

cube, hdrbuffer, blurbuffer, lastPosX, lastPosY = None, None, None, None, None
# Default voxel presentation, options:
# generic: no outlier voxels, generic coloring
# generic_outliers: includes outlier voxels, generic coloring
# color: no outlier voxels, camera view coloring
# color_outliers: includes outlier voxels, camera view coloring
# tracking: no outlier voxels, cluster coloring for tracking
defaultVisual = "tracking"
# Generated voxel positions and colors for presentation
positions, colors, colorsCamera, colorsTracking, positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers = \
    None, None, None, None, None, None, None
# Flag to indicate if looping voxel generation process
isLooping = False
# Flags for screenshots, if True then takes a screenshot for every frame change
takeScreenshots = True
screenshotFrame = 0
sceneRendered = False

firstTime = True
window_width, window_height = config['window_width'], config['window_height']
camera = Camera(glm.vec3(0, 100, 0), pitch=-90, yaw=0, speed=40)


def take_window_screenshot(window, screenshot_output_path="ss", screenshot_output_filename="screenshot.png"):
    """
    Takes a screenshot of the open execution window and outputs it.

    :param window: execution window
    :param screenshot_output_path: screenshot output directory path
    :param screenshot_output_filename: screenshot output file name (including extension)
    """
    # Take screenshot
    width, height = glfw.get_framebuffer_size(window)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    # OpenGL stores images upside down
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save screenshot
    image.save(os.path.join(screenshot_output_path, screenshot_output_filename))


def compile_window_screenshots_into_video(screenshot_input_path="ss", video_output_filename="compiled_shots.mp4"):
    """
    Compiles screenshots of the open execution window into a video and outputs it.

    :param screenshot_input_path: screenshot input directory path
    :param video_output_filename: video output file name (including extension), file is saved in the same path as
                                  screenshot_input_path
    """
    # Get screenshot files
    image_files = [os.path.join(screenshot_input_path, file) for file in os.listdir(screenshot_input_path)
                   if file.startswith("shot") and file.endswith(".png")]
    # Sort files (avoid case like shot2 being after shot11)
    image_files.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)])

    # Read the first image to get the dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the codec for MP4 and create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_cap = cv2.VideoWriter(os.path.join(screenshot_input_path, video_output_filename), fourcc, 1,
                                 (width, height))

    # Write each image to the video writer
    for image_file in image_files:
        image = cv2.imread(image_file)
        output_cap.write(image)
    output_cap.release()


def delete_screenshots_and_videos(file_path="ss"):
    """
    Deletes taken screenshots and compiled videos.

    :param file_path: directory path
    """
    # Select screenshots and videos
    image_files = [os.path.join(file_path, file) for file in os.listdir(file_path)
                   if file.startswith("shot") and file.endswith(".png")]
    video_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(".mp4")]
    files = image_files + video_files

    # Delete files
    for file in files:
        os.remove(file)


def draw_objs(obj, program, perspective, light_pos, texture, normal, specular, depth):
    program.use()
    program.setMat4('viewProject', perspective * camera.get_view_matrix())
    program.setVec3('viewPos', camera.position)
    program.setVec3('light_pos', light_pos)

    glActiveTexture(GL_TEXTURE1)
    program.setInt('mat.diffuseMap', 1)
    texture.bind()

    glActiveTexture(GL_TEXTURE2)
    program.setInt('mat.normalMap', 2)
    normal.bind()

    glActiveTexture(GL_TEXTURE3)
    program.setInt('mat.specularMap', 3)
    specular.bind()

    glActiveTexture(GL_TEXTURE4)
    program.setInt('mat.depthMap', 4)
    depth.bind()
    program.setFloat('mat.shininess', 128)
    program.setFloat('mat.heightScale', 0.12)

    obj.draw_multiple(program)


def main():
    global hdrbuffer, blurbuffer, cube, window_width, window_height, \
        positions, colors, colorsCamera, colorsTracking, \
        positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers, \
        isLooping, takeScreenshots, screenshotFrame, sceneRendered

    if not glfw.init():
        print('Failed to initialize GLFW.')
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, config['sampling_level'])

    if config['fullscreen']:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        window_width, window_height = mode.size.window_width, mode.size.window_height
        window = glfw.create_window(mode.size.window_width,
                                    mode.size.window_height,
                                    config['app_name'],
                                    glfw.get_primary_monitor(),
                                    None)
    else:
        window = glfw.create_window(window_width, window_height, config['app_name'], None, None)
    if not window:
        print('Failed to create GLFW Window.')
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_framebuffer_size_callback(window, resize_callback)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    depth_program = get_linked_program('resources/shaders/shadow_depth.vs', 'resources/shaders/shadow_depth.fs')
    blur_program = get_linked_program('resources/shaders/blur.vs', 'resources/shaders/blur.fs')
    hdr_program = get_linked_program('resources/shaders/hdr.vs', 'resources/shaders/hdr.fs')

    blur_program.use()
    blur_program.setInt('image', 0)

    hdr_program.use()
    hdr_program.setInt('sceneMap', 0)
    hdr_program.setInt('bloomMap', 1)

    window_width_px, window_height_px = glfw.get_framebuffer_size(window)

    hdrbuffer = HDRBuffer()
    hdrbuffer.create(window_width_px, window_height_px)
    blurbuffer = BlurBuffer()
    blurbuffer.create(window_width_px, window_height_px)

    bloom = Bloom(hdrbuffer, hdr_program, blurbuffer, blur_program)

    light_pos = glm.vec3(0.5, 0.5, 0.5)
    perspective = glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])

    cam_rot_matrices = get_cam_rotation_matrices()
    cam_shapes = [Model('resources/models/camera.json', cam_rot_matrices[c]) for c in range(4)]
    square = Model('resources/models/square.json')
    cube = Model('resources/models/cube.json')
    texture = load_texture_2d('resources/textures/diffuse.jpg')
    texture_grid = load_texture_2d('resources/textures/diffuse_grid.jpg')
    normal = load_texture_2d('resources/textures/normal.jpg')
    normal_grid = load_texture_2d('resources/textures/normal_grid.jpg')
    specular = load_texture_2d('resources/textures/specular.jpg')
    specular_grid = load_texture_2d('resources/textures/specular_grid.jpg')
    depth = load_texture_2d('resources/textures/depth.jpg')
    depth_grid = load_texture_2d('resources/textures/depth_grid.jpg')

    grid_positions, grid_colors = generate_grid(config['world_width'], config['world_width'])
    square.set_multiple_positions(grid_positions, grid_colors)

    cam_positions, cam_colors = get_cam_positions()
    for c, cam_pos in enumerate(cam_positions):
        cam_shapes[c].set_multiple_positions([cam_pos], [cam_colors[c]])

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        if config['debug_mode']:
            print(glGetError())

        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        move_input(window, delta_time)

        # Loop for frames if L was pressed during key_callback
        if isLooping:
            positions, colors, colorsCamera, colorsTracking, \
                positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers, isFinished \
                = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])

            # tracking: no outlier voxels, cluster coloring for tracking
            if defaultVisual == "tracking":
                cube.set_multiple_positions(positions, colorsTracking)
            # color: no outlier voxels, camera view coloring
            elif defaultVisual == "color":
                cube.set_multiple_positions(positions, colorsCamera)
            # color_outliers: includes outlier voxels, camera view coloring
            elif defaultVisual == "color_outliers":
                cube.set_multiple_positions(positionsWithOutliers, colorsCameraWithOutliers)
            # generic: no outlier voxels, generic coloring
            elif defaultVisual == "generic":
                cube.set_multiple_positions(positions, colors)
            # generic_outliers: includes outlier voxels, generic coloring
            else:
                cube.set_multiple_positions(positionsWithOutliers, colorsWithOutliers)

            # Reset and stop looping if finished
            if isFinished:
                positions, colors, colorsCamera, colorsTracking, \
                    positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers = \
                    None, None, None, None, None, None, None
                isLooping = False
                sceneRendered = False

                # Combine screenshots into video
                compile_window_screenshots_into_video("ss", "compiled_shots.mp4")
            else:
                sceneRendered = True

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.2, 0.8, 1)

        square.draw_multiple(depth_program)
        cube.draw_multiple(depth_program)
        for cam in cam_shapes:
            cam.draw_multiple(depth_program)

        hdrbuffer.bind()

        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        glViewport(0, 0, window_width_px, window_height_px)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_objs(square, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)
        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        for cam in cam_shapes:
            draw_objs(cam, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        hdrbuffer.unbind()
        hdrbuffer.finalize()

        bloom.draw_processed_scene()

        # Screenshot after first visualization of voxels
        if sceneRendered and takeScreenshots:
            take_window_screenshot(window, "ss", "shot" + str(screenshotFrame) + ".png")
            screenshotFrame += 1
            sceneRendered = False

        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


def resize_callback(window, w, h):
    if h > 0:
        global window_width, window_height, hdrbuffer, blurbuffer
        window_width, window_height = w, h
        glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])
        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        hdrbuffer.delete()
        hdrbuffer.create(window_width_px, window_height_px)
        blurbuffer.delete()
        blurbuffer.create(window_width_px, window_height_px)


def key_callback(window, key, scancode, action, mods):
    global cube, defaultVisual, \
        positions, colors, colorsCamera, colorsTracking, \
        positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers, \
        isLooping, takeScreenshots, screenshotFrame, sceneRendered

    # Exit window
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, glfw.TRUE)

    # Generate next frame voxels
    if key == glfw.KEY_G and action == glfw.PRESS:
        positions, colors, colorsCamera, colorsTracking, \
            positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers, isFinished \
            = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])

        # tracking: no outlier voxels, cluster coloring for tracking
        if defaultVisual == "tracking":
            cube.set_multiple_positions(positions, colorsTracking)
        # color: no outlier voxels, camera view coloring
        elif defaultVisual == "color":
            cube.set_multiple_positions(positions, colorsCamera)
        # color_outliers: includes outlier voxels, camera view coloring
        elif defaultVisual == "color_outliers":
            cube.set_multiple_positions(positionsWithOutliers, colorsCameraWithOutliers)
        # generic: no outlier voxels, generic coloring
        elif defaultVisual == "generic":
            cube.set_multiple_positions(positions, colors)
        # generic_outliers: includes outlier voxels, generic coloring
        else:
            cube.set_multiple_positions(positionsWithOutliers, colorsWithOutliers)

        # Reset if finished
        if isFinished:
            positions, colors, colorsCamera, colorsTracking,\
                positionsWithOutliers, colorsWithOutliers, colorsCameraWithOutliers = \
                None, None, None, None, None, None, None
            sceneRendered = False

            # Combine screenshots into video
            compile_window_screenshots_into_video("ss", "compiled_shots.mp4")
        else:
            sceneRendered = True

    # Loop over remainder of frames in main loop
    if key == glfw.KEY_L and action == glfw.PRESS:
        isLooping = True

    # Change visual
    if key == glfw.KEY_1 and action == glfw.PRESS:
        # Show generic_outliers: includes outlier voxels, generic coloring
        if positions is not None:
            cube.set_multiple_positions(positionsWithOutliers, colorsWithOutliers)
        # Set to default for next generation for continuity
        defaultVisual = "generic_outliers"
    if key == glfw.KEY_2 and action == glfw.PRESS:
        # Show color_outliers: includes outlier voxels, camera view coloring
        if positions is not None:
            cube.set_multiple_positions(positionsWithOutliers, colorsCameraWithOutliers)
        # Set to default for next generation for continuity
        defaultVisual = "color_outliers"
    if key == glfw.KEY_3 and action == glfw.PRESS:
        # Show generic: no outlier voxels, generic coloring
        if positions is not None:
            cube.set_multiple_positions(positions, colors)
        # Set to default for next generation for continuity
        defaultVisual = "generic"
    if key == glfw.KEY_4 and action == glfw.PRESS:
        # Show color: no outlier voxels, camera view coloring
        if positions is not None:
            cube.set_multiple_positions(positions, colorsCamera)
        # Set to default for next generation for continuity
        defaultVisual = "color"
    if key == glfw.KEY_5 and action == glfw.PRESS:
        # Show tracking: no outlier voxels, cluster coloring for tracking
        if positions is not None:
            cube.set_multiple_positions(positions, colorsTracking)
        # Set to default for next generation for continuity
        defaultVisual = "tracking"


def mouse_move(win, pos_x, pos_y):
    global firstTime, camera, lastPosX, lastPosY
    if firstTime:
        lastPosX = pos_x
        lastPosY = pos_y
        firstTime = False

    camera.rotate(pos_x - lastPosX, lastPosY - pos_y)
    lastPosX = pos_x
    lastPosY = pos_y


def move_input(win, time):
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        camera.move_top(time)
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        camera.move_bottom(time)
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        camera.move_left(time)
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        camera.move_right(time)


if __name__ == '__main__':
    delete_screenshots_and_videos("ss")
    main()
