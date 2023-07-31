cmake_minimum_required(VERSION 2.8.3)
project(bumpypatch)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation  # Add this line
)

## Generate added messages and services with any dependencies listed here
# add_message_files(
#    FILES
#    BumpyPatchInfo.msg
# )

# add_service_files(
#   FILES
#   SaveImage.srv
# )
## Generate added messages and services with any dependencies listed here
# generate_messages(
#    DEPENDENCIES
#    std_msgs
# )

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a catkin package
catkin_package(
  LIBRARIES bumpypatch
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime  # Add 'message_runtime' here
)

## Build nodes

catkin_install_python(PROGRAMS src/bumpypatch.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})




