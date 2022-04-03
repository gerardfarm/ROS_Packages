# ROS_Packages
Implememtation of differerent ROS packages for autonomous robot navigation

## Getting started
put the packages: perception, localization and custom messages in projectFolder/src.

```
colcon build
. install/setup.bash
```

```
ros2 run perception cam_publisher   # to get images from streaming
ros2 run perception detect_objects  # to take image and compute bounding boxes
ros2 run perception bbox_subscriber # to get bounding boxes
```

```
ros2 run localization gps_publisher  # to get GPS data
ros2 run localization acc_publisher  # to get Acc data
ros2 run localization gyro_publisher # to get Gyro data
ros2 run localization localize_robot # taking all sensors data and run EKF to estimate states (position, velocity and orientation)
```
