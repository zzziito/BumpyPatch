cmake_minimum_required(VERSION 3.0.2)
project(bumpypatch)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs jsk_recognition_msgs)

## Declare a catkin package
catkin_package(
    CATKIN_DEPENDS rospy std_msgs sensor_msgs
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

## Specify additional locations of python modules
catkin_python_setup()

install(PROGRAMS
  src/bumpypatch_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
