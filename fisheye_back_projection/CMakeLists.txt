cmake_minimum_required(VERSION 2.8.3)
project(fisheyes_backprojection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp roslib std_msgs geometry_msgs tf message_generation)
find_package(cmake_modules REQUIRED)
find_package(OpenCV)
find_package(Eigen3)

add_message_files(
  DIRECTORY msg
  FILES
  PredictionStamped.msg
  ReprojectionArrayStamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package()

include_directories(include "/opt/ros/melodic/include" ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${tf_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_executable(point_backprojection src/point_backprojection.cpp)
target_link_libraries(point_backprojection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(point_backprojection controllers_generate_messages_cpp controllers_gencfg)