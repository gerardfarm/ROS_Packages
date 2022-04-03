# ROS_Packages
Implememtation of differerent ROS packages for autonomous robot navigation

## Getting started
Put the packages: perception, localization and custom messages in projectFolder/src.

First, you need to build packages:
```
colcon build
. install/setup.bash
```

In perception package, you need to setup the object detection library by running the following commands:
```
cd src/perception/perception
sudo python3 setup.py install
```
Then, you can test this package on your webcam:
```
ros2 run perception cam_publisher   # to get images from streaming
ros2 run perception detect_objects --ros-args -p source:='0' -p visualize:=0
ros2 run perception bbox_subscriber # to get bounding boxes
```

To work on the localization part, run the following instructions:
```
ros2 run localization gps_publisher  # to get GPS data
ros2 run localization acc_publisher  # to get Acc data
ros2 run localization gyro_publisher # to get Gyro data
ros2 run localization localize_robot # taking all sensors data and run EKF to estimate states (position, velocity and orientation)
```
