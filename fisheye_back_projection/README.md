# ROS Package for Gate Backprojection

ROS Package for Gate Backprojection

## Frame annotation

We use image and camera frame with the same convention like ROS calibration package and OpenCV (right-handed, Z inwards, X to the right and Y downwards looking toward +Z). Please see [here](http://wiki.ros.org/image_pipeline/CameraInfo)

TODO: a graph explaining frame transformation needed

## Camera calibration

This package uses camera calibration information (using ROS format) to retrieve projection and reprojection matrix (through the config/.yaml file). When using for resized input image, please also re-calculate the .yaml file accordingly.

## Limitations

- The package does not take account of distortion in the calculation.