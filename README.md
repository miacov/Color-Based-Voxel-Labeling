# Color-Based Voxel Labeling
Voxel-Based 3D Object Reconstruction and Figure Tracking using Color-Based Voxel Labeling

![Results](/ss/final_video.gif)

## Control
Run `executable.py` and use buttons:

* `G` to render the next frame
* `L` to render all the remaining frames
* `ESC` to exit the program

To switch the type of rendering, use buttons:
* `1` to show voxels with outliers using generic coloring
* `2` to show voxels with outliers using coloring from camera 2
* `3` to show voxels without outliers using generic coloring
* `4` to show voxels without outliers using coloring from camera 2
* `5` to show voxels without outliers using coloring from color model labels (default option)

## Camera Calibration
The calibration of the cameras is done in `camera_calibration.py` using provided data for every camera found in the `cam1`, `cam2`, `cam3`, and `cam4` folders under the `data` folder. Chessboard measurements are found in `checkerboard.xml` under the same folder. The parameters for each camera are saved in `config.xml` under every camera folder. Plots for camera calibration can be found under `plots`.

![Calibration evaluation](/data/cam2/test.jpg)

The intrinsic parameters are calculated using video frames from `intrinsics.avi`, which is found under every camera folder. Only frames where OpenCV successfully extracts corners are kept for the calibration. From that frame selection, frames that negatively affect the calibration results are discarded using an iterative discarding algorithm.

The extrinsic parameters are calculated using a video frame from `checkerboard.avi`, which is found under every camera folder. OpenCV fails to extract corners from frames in these videos and a different algorithm is implemented to find the corners automatically. In case the results of this algorithm are not accurate, the extracted corners can be discarded and new corners can be manually selected instead.

![Corner detection](/data/cam2/checkerboard_imagepoints.jpg)

## Background Subtraction
Different background subtractors are tested in `background_subtraction.py`, namely KNN (K-Nearest Neighbors), MOG (Mixture of Gaussians) and MOG2. These subtractors are trained on video frames from `background.avi`, which is found  under every camera folder. Further thresholding with contour detection and morphological operations is applied on the foreground masks extracted from the background subtractors to improve the results. Plots comparing the resulting foreground masks can be found under `plots`.

![Foreground mask](/data/cam2/mask_MOG2.jpg)

## Voxel Reconstruction
A 128x64x128 (width x height x depth) voxel space with a spacing of 60 units is created for voxel reconstruction in `assignment.py` using functions from `voxel_reconstruction.py`. Each of these 3D voxel points are projected to the image plane using the intrinsic and extrinsic parameters of each camera to create a lookup table. The calibration parameters are also used to position and rotate the cameras in the 3D scene.

During the execution of `executable.py`, every time a new frame is called to be rendered, the next frame from every `video.avi`, which is found under every camera folder, is extracted. An interval of 5 frames is used to progress through frames where minimal changes are observed. The extracted frames are camera views during the current timestamp. For each of these views, a foreground mask is extracted using a trained MOG2 background subtractor and additional thresholding. Voxels projected onto the foreground mask of every one of the extracted frames are marked as visible for the current timestamp, while the rest are turned off.

![Clusters](/plots/marching_cubes_front.png)

The visible voxels are clustered using k-means clustering by taking into account just their width and depth dimensions and ignoring their height dimension. The number of clusters is equal to the number of figures in the videos, which is 4. Distances from cluster centroids are used to remove some potential outlier and ghost voxels.

![Clusters](/plots/voxel_clusters_online.png)

The clustered voxels are used to create color models of every figure in the videos. Offline color models for every camera view are computed using the first frame of every video. The color models are HSV histograms of the pixels corresponding to each figure when projecting the 3D clustered voxel points to the image plane. Cropping is used to focus the color models on the shirts of each figure.

![Offline color models](/data/cam2/video_color_model_projections.jpg)

During execution, online color models are computed using the current frame of every video. In order to match these color models to the corresponding offline ones, a distance matrix of every online color model’s similarity to each offline model is computed. The similarity is calculated by comparing the histograms of the color models across camera views and color channels. The method for comparison is Bhattacharyya distance in order for similiarity to be close to a 0 value. The Hungarian algorithm is run on the distance matrix to match the online color models to the offline ones. The voxels belonging to each online color model cluster are labeled with the color originally given to the offline color model clusters. More details about the color models can be found in `report.pdf`.

![Online color models](/ss/angle1/shot0.png)

Using the labels of the color models, each figure in the video is tracked. A trajectory image is created with the cluster centers of each figure at each timestamp of the videos. These trajectories are also saved under `data`.

![Trajectories](/plots/trajectories_raw.png)

To close the gaps in the trajectories and make them continuous, the trajectory points are interpolated using a set number of new samples. The new samples are set to 5 times the number of frames divided by the frame interval.

![Smoothed trajectories](/plots/trajectories_smooth.png)

Screenshots and videos of the results of voxel reconstruction and figure tracking can be found under `ss`. Plots of visible voxels converted to a surface mesh using Marching Cubes, voxel clusters, and figure trajectories can be found under `plots`.

## Codebase
* https://github.com/stanfortonski/Perlin-Noise-3D-Voxel-Generator
* https://github.com/dmetehan/Computer-Vision-3D-Reconstruction
