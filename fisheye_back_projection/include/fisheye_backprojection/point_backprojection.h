#ifndef POINT_BACKPROJECTION_H
#define POINT_BACKPROJECTION_H

#include <ros/ros.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <std_msgs/Float64.h>
#include "opencv2/opencv.hpp"
#include <fisheyes_backprojection/PredictionStamped.h>
#include <fisheyes_backprojection/ReprojectionArrayStamped.h>
// #include <perception_module/PredictionStamped.h>

#include <Eigen/Dense>

#define MIN_PROJECT_DISTANCE 2.0  // minimum trustred prediction range
#define MAX_PROJECT_DISTANCE 5.0  // maximum trusted prediction range
#define MIN_GOOD_DISTANCE 2.5//2.8  // maximum trusted prediction range
#define MAX_GOOD_DISTANCE 3.0//3.3  // maximum trusted prediction range


using namespace std;
using namespace ros;
using namespace cv;


class PointBackprojection{
  public:
    PointBackprojection(int, char**, ros::NodeHandle& nh_input);
    ~PointBackprojection();
    // void run();
    Mat backprojectPoint(Point2d point, double distance);
    Mat backprojectPointFishEyeCam(Point2d point, double distance); //back project for fish-eye

    void pointsCallback(const geometry_msgs::PolygonStamped& points_msg);
    void predictionCallback(const fisheyes_backprojection::PredictionStamped& prediction_msg);
  
  private:

    // Publishers
    ros::Publisher points_publisher, prediction_publisher; 
    ros::Publisher position_frame_d_publisher;  // for MSF

    // Subscribers
    ros::Subscriber points_subscriber;  // will be deprecated in future release. Not stable by this time
    ros::Subscriber prediction_subscriber;  // if there is orientation. This should be the default
    ros::Subscriber vicon_subscriber;

    // Convert gate predicted in drone's body frame to world frame (using TF)
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
    geometry_msgs::TransformStamped transformStamped_;


    Mat projectionMatrix, inverseProjectionMatrix;

    Mat camera_matrix_;


    ros::NodeHandle node_handle_;

    std::vector<double> projectionMatrixParams;

};

#endif /* POINT_BACKPROJECTION_H */