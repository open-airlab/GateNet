

# GateNet: A Shallow Neural Network For Gate Perception In Drone Racing
This repo is a part of the complement material of the paper GateNet: Efficient Deep Neural Network for Gate Perception in Autonomous Drone Racing.

The content is limited since the paper in under review.


## GateNet

The figure illustrates  of the working principles of our gate perception system. The scene images are captured by a single wide-FOV fish-eye RBG camera and fed into GateNet to estimate the gate center location, as well as distance and orientation with respect to the drone's body frame. The information then will be used to re-project the gate in 3D world frame and applied an extended Kalman Filter to achieve stable gate pose estimation.

![alt text](https://raw.githubusercontent.com/open-airlab/GateNet/master/visual_abstract.png) 

### AU-DR Dataset
The AU-DR dataset includes different gate layout cases that appear in a drone racing scenario: (a) single gate, (b)multiple gates, (c ) occluded gates, (d) partially observable gates, (e) gates with a distant layout, and (f) a gate that is too close to the drone’s camera.
 
![alt text](https://raw.githubusercontent.com/open-airlab/GateNet/master/samples.png)
 
 You can download AU-DR datasets [here](https://drive.google.com/file/d/1tkLJri7lnPIcUq93XHbT8sByT6V3ynA2/view?usp=sharing).

## References

H.Phum, I. Bozcan, A. Sarabakha, S. Haddadin and E. Kayacan, "GateNet: Efficient Deep Neural Network for Gate Perception in Autonomous Drone Racing", submitted to IEEE International Conference on Robotics and Automation (ICRA) 2021.


