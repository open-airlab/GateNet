#include "fisheye_backprojection/point_backprojection.h"

// Constructor
PointBackprojection::PointBackprojection(int argc, char** argv, ros::NodeHandle& nh_input)
    : node_handle_(nh_input), projectionMatrix(Mat_<double>(3,4)), inverseProjectionMatrix(Mat_<double>(4,3)), tfListener_(tfBuffer_)
{

    if(node_handle_.getParam("/projection_matrix/data", projectionMatrixParams)){
        memcpy(projectionMatrix.data, projectionMatrixParams.data(), projectionMatrixParams.size() * sizeof(double));
        // cout << "[PointBackprojection] Projection Matrix: " << endl << projectionMatrix << endl << endl;

        invert(projectionMatrix, inverseProjectionMatrix, DECOMP_SVD);
        // cout << "[PointBackprojection] Inverse Projection Matrix: " << endl << inverseProjectionMatrix << endl << endl;
    }
    else
        ROS_ERROR("Failed to get param '/projection_matrix/data'");

    points_subscriber = node_handle_.subscribe("/gates/detected_2D", 1, &PointBackprojection::pointsCallback, this);
    
    points_publisher = node_handle_.advertise<geometry_msgs::PolygonStamped>("/gates/backprojected_3D", 1);


    // fish eye
    prediction_subscriber = node_handle_.subscribe("/detection_results", 1, &PointBackprojection::predictionCallback, this);

    // prediction_publisher  = node_handle_.advertise<geometry_msgs::PoseArray>("/gates/gate_array_reprojected", 1);
    prediction_publisher  = node_handle_.advertise<fisheyes_backprojection::ReprojectionArrayStamped>("/gates/gate_array_reprojected", 1);

    // position_frame_d_publisher  = node_handle_.advertise<geometry_msgs::PointStamped>("/gates/position_3D_in_drone_frame", 1);


    camera_matrix_ = (Mat_<double>(3,3) << 635.3984250929781/9, 0, 715.4594697095278/9, 0, 635.3992921038243/9, 542.0308149398195/9, 0, 0, 1);

}

// Destructor
PointBackprojection::~PointBackprojection(){
    ros::shutdown();
    exit(0);
}


Mat PointBackprojection::backprojectPoint(Point2d point, double distance){  //back project point from image 2D coordinate + distance to camera frame 3D
    Mat point2D = (Mat_<double>(3,1) << point.x, point.y, 1);
    //cout << "[PointReprojection] 2D Point: " << point2D.t() << endl << endl;

    Mat backprojectedPoint = inverseProjectionMatrix * point2D;
    //cout << "[PointReprojection] Backprojected Point: " << backprojectedPoint.t() << endl << endl;

    double scale = distance / sqrt(pow(backprojectedPoint.at<double>(0,0), 2) + pow(backprojectedPoint.at<double>(0,1), 2) + pow(backprojectedPoint.at<double>(0,2), 2));

    Mat backprojectedDenormalizedPoint = backprojectedPoint * scale;
    //cout << "[PointReprojection] Backprojected Point: " << backprojectedDenormalizedPoint.t() << endl << endl;

    return backprojectedDenormalizedPoint;
}


Mat PointBackprojection::backprojectPointFishEyeCam(Point2d point, double distance){  //back project point from image 2D coordinate + distance to camera frame 3D
    
    Mat point2D = (Mat_<double>(3,1) << point.x, point.y, 1);

    // Mat distortion_coeffs = (Mat_<double>(1,5) << 1, 0.007258083582203451, -0.0024109130635207566, 0.00399062145907147, -0.0014809788398292363);
    Eigen::VectorXd distortion_coeffs(5);
    distortion_coeffs << 1, 0.007258083582203451, -0.0024109130635207566, 0.00399062145907147, -0.0014809788398292363;
    // transpose(distortion_coeffs, distortion_coeffs);
    // Mat backprojectedPoint = inverseProjectionMatrix * point2D;

    Mat points_distorted = camera_matrix_.inv() * point2D;
    
    double theta_d = sqrt(pow(points_distorted.at<double>(0,0), 2) + pow(points_distorted.at<double>(0,1), 2));
    double theta = theta_d;


    for (int i = 0; i < 10; i ++){
        // Mat theta_nominator = (Mat_<double>(5,1) << theta, pow(theta,3), pow(theta,5), pow(theta,7), pow(theta,9) );
        // Mat theta_denominator = (Mat_<double>(5,1) << 1, 3*pow(theta,2), 5*pow(theta,4), 7*pow(theta,6), 8*pow(theta,8) );

        Eigen::VectorXd theta_nominator(5);
        theta_nominator << theta, pow(theta,3), pow(theta,5), pow(theta,7), pow(theta,9);
        Eigen::VectorXd theta_denominator(5);
        theta_denominator << 1, 3*pow(theta,2), 5*pow(theta,4), 7*pow(theta,6), 8*pow(theta,8);

        double d_theta = (theta_d -  distortion_coeffs.transpose() * theta_nominator)
                                / (distortion_coeffs.transpose() * theta_denominator);

        // double d_theta = (theta_d - ( theta + pow(theta,3) + pow(theta,5) + pow(theta,7) + pow(theta,9) ) ) /
        //                 ( 1 + 3*pow(theta,2) + 5*pow(theta,4) + 7*pow(theta,6) + 8*pow(theta,8) );        
        theta = theta + d_theta;
        if (fabs(d_theta) < 1e-9)
            break;
    }
    
    //cout << "[PointReprojection] 2D Point: " << point2D.t() << endl << endl;

    double scale = tan(theta)/theta_d;
    Mat points_undistorted = points_distorted * scale;

    points_undistorted.at<double>(0,2) = 1;


    Mat backprojectedDenormalizedPoint = points_undistorted * distance 
                                        / sqrt(pow(points_undistorted.at<double>(0,0), 2) + pow(points_undistorted.at<double>(0,1), 2) + pow(points_undistorted.at<double>(0,2), 2)); 

    return backprojectedDenormalizedPoint;
}

void PointBackprojection::predictionCallback(const fisheyes_backprojection::PredictionStamped& prediction_msg){
    int start_s=clock();

    fisheyes_backprojection::ReprojectionArrayStamped points_backprojected_msg;
    // geometry_msgs::PoseArray points_backprojected_msg;
    Mat backprojectedPoint;

    points_backprojected_msg.header = prediction_msg.header;

    if (prediction_msg.predicted_center_x.size() > 0) {

        for (int i = 0; i < prediction_msg.predicted_center_x.size(); i ++){

            double pixel_cx = prediction_msg.predicted_center_x[i].data;
            double pixel_cy = prediction_msg.predicted_center_y[i].data;
            double predicted_distance = prediction_msg.predicted_distance[i].data;
            double predicted_yaw = prediction_msg.predicted_yaw[i].data;

            if ( predicted_distance >= MIN_PROJECT_DISTANCE && predicted_distance <= MAX_PROJECT_DISTANCE) {  // check if prediction is trusted
                // std::cout << "cx = " << pixel_cx << ", cy = " << pixel_cy << ", distance = " << predicted_distance << std::endl;
                // int start_s1=clock();
                backprojectedPoint = backprojectPointFishEyeCam(Point2d(pixel_cx, pixel_cy), predicted_distance);
                // int stop_s1=clock();
                // std::cout << " reprojection time = " << (stop_s1-start_s1)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;

                // Convert from camera frame to drone's body frame (right handed system)

                // from camera image to camera frame
                Eigen::Vector3d gate_predicted_camera_frame;
                gate_predicted_camera_frame << backprojectedPoint.at<double>(0,2), -backprojectedPoint.at<double>(0,0), -backprojectedPoint.at<double>(0,1);
                
                // Transform to drone's frame
                Eigen::Matrix3d rotation_mat;
                rotation_mat = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ())   
                                * Eigen::AngleAxisd(-0.3142, Eigen::Vector3d::UnitY())   // camera is pitched - 18 deg (-0.3142 rad) from the body x-axis
                                * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());   // previously 0
                
                Eigen::Vector3d translation_camera_offset;
                translation_camera_offset << 0.05, 0.00, 0.01;  // translation offset from the camera to drone COM
                
                
                Eigen::Vector3d gate_predicted_drone_frame = rotation_mat * gate_predicted_camera_frame + translation_camera_offset;
                
                // wrap-up for topics
                geometry_msgs::PoseStamped gate_drone_frame;
                gate_drone_frame.header.stamp = prediction_msg.header.stamp;
                gate_drone_frame.header.frame_id = "base_link";
                gate_drone_frame.pose.position.x = gate_predicted_drone_frame(0);
                gate_drone_frame.pose.position.y = gate_predicted_drone_frame(1);
                gate_drone_frame.pose.position.z = gate_predicted_drone_frame(2);

                tf2::Quaternion q_rot;
                q_rot.setRPY(0, 0,predicted_yaw);
                tf2::convert(q_rot, gate_drone_frame.pose.orientation);


                // Convert gate predicted in drone's body frame to world frame (using TF)
                geometry_msgs::PoseStamped tf2_gate_world_frame;
                try {
                    // transformStamped_ = tfBuffer_.lookupTransform("world", "vicon/ida_racer/ida_racer", ros::Time(0), ros::Duration(0.05));   //target frame, source frame
                    transformStamped_ = tfBuffer_.lookupTransform("map", "drone_base_link", ros::Time(0), ros::Duration(0.05));   //target frame, source frame
                    tf2::doTransform(gate_drone_frame, tf2_gate_world_frame, transformStamped_);    // (input, output, transform)
                }
                catch (tf2::TransformException &ex) {ROS_WARN("%s",ex.what());
                }  // ;ros::Duration(1.0).sleep(); continue;

                // geometry_msgs::Pose pose;
                // pose.position = tf2_gate_world_frame.pose.position;
                // pose.orientation = tf2_gate_world_frame.pose.orientation;
                // // Publish reprojected point in world frame
                // points_backprojected_msg.poses.push_back(pose);

                geometry_msgs::PoseWithCovariance pose_with_cov;
                
                pose_with_cov.pose.position = tf2_gate_world_frame.pose.position;
                pose_with_cov.pose.orientation = tf2_gate_world_frame.pose.orientation;
                
                if ( predicted_distance >= MIN_GOOD_DISTANCE && predicted_distance <= MAX_GOOD_DISTANCE ){  // in reliable range
                    pose_with_cov.covariance[0] = 0.5;  // confidence in position (0.5)
                    pose_with_cov.covariance[7] = 0.5;
                    pose_with_cov.covariance[14] = 0.5; 
                    pose_with_cov.covariance[21] = 1.0; // confidence in orientation (0.5)
                    pose_with_cov.covariance[28] = 1.0;
                    pose_with_cov.covariance[35] = 1.0;
                }
                else{
                    pose_with_cov.covariance[0] = 1.0;
                    pose_with_cov.covariance[7] = 1.0;
                    pose_with_cov.covariance[14] = 1.0;
                    pose_with_cov.covariance[21] = 1.0;
                    pose_with_cov.covariance[28] = 1.0;
                    pose_with_cov.covariance[35] = 1.0;                    
                }
                // Publish reprojected point in world frame
                points_backprojected_msg.poses_with_cov_matrices.push_back(pose_with_cov);
            }

            
        }

    }

    prediction_publisher.publish(points_backprojected_msg);

    int stop_s=clock();
    std::cout << "[Backprojection_Node] total time = " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;

}


void PointBackprojection::pointsCallback(const geometry_msgs::PolygonStamped& points_msg){
    geometry_msgs::PolygonStamped points_backprojected_msg;
    geometry_msgs::Point32 point;
    Mat backprojectedPoint;

    for(int i = 0; i < points_msg.polygon.points.size(); ++i){
        backprojectedPoint = backprojectPoint(Point2d(points_msg.polygon.points[i].x, points_msg.polygon.points[i].y), points_msg.polygon.points[i].z);

        // points_backprojected_msg.polygon.points.push_back(point);    //ToDO: support reprojection of more than one point.
    }

    geometry_msgs::PointStamped gate_predicted_drone_frame;

    // Convert from camera frame to drone's body frame (right handed system)
    gate_predicted_drone_frame.header.stamp = ros::Time::now();
    gate_predicted_drone_frame.header.frame_id = "base_link";
    gate_predicted_drone_frame.point.x = backprojectedPoint.at<double>(0,2);;
    gate_predicted_drone_frame.point.y = -backprojectedPoint.at<double>(0,0);
    gate_predicted_drone_frame.point.z = -backprojectedPoint.at<double>(0,1);

    // Publish gate in drone frame (for msf)
    // position_frame_d_publisher.publish(gate_predicted_drone_frame);

    geometry_msgs::Point32 gate_predicted_world_frame;
    geometry_msgs::PointStamped tf2_gate_predicted_world_frame;

    try{
        transformStamped_ = tfBuffer_.lookupTransform("world", "/vicon/ida_racer/ida_racer", ros::Time(0), ros::Duration(0.5));   //target frame, source frame
        tf2::doTransform(gate_predicted_drone_frame, tf2_gate_predicted_world_frame, transformStamped_);    // (input, output, transform)
        gate_predicted_world_frame.x = tf2_gate_predicted_world_frame.point.x;
        gate_predicted_world_frame.y = tf2_gate_predicted_world_frame.point.y;
        gate_predicted_world_frame.z = tf2_gate_predicted_world_frame.point.z;
    }
    catch (tf2::TransformException &ex) {ROS_WARN("%s",ex.what());
    }  // ;ros::Duration(1.0).sleep(); continue;

    // Publish reprojected point in world frame
    points_backprojected_msg.polygon.points.push_back(gate_predicted_world_frame); //put point in world
    points_backprojected_msg.header.stamp = Time::now();
    points_publisher.publish(points_backprojected_msg);

}

int main(int argc, char** argv){
    // Initialize ROS
    ros::init (argc, argv, "PointReprojection");
    ros::NodeHandle nh;
    cout << "[Backprojection_Node] Fisheye Point Backprojection is running..." << endl;

    PointBackprojection* pb = new PointBackprojection(argc, argv, nh);

    // pb->run();

    // Spin
    ros::spin ();
    return 0;
}

