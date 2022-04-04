# ROS_Packages
Implememtation of differerent ROS packages for autonomous robot navigation

## Getting started
Put the packages: perception, localization and custom messages in projectFolder/src.

First, you need to build packages:
```
colcon build
. install/setup.bash
```
Then to run the whole project, you can simply run:
```
ros2 launch brain brain.launch.py
```

For more details about each package, see the following sections.

### Custom_msgs
Nothing should be run in this package. This one is used to define our sent and received messages between different nodes.

### Perception
The ```perception``` package is used to detect objects in a streaming and publish necessary informations about bounding boxes (coordinates, class label, confidence score).

In this package, you need to setup the object detection library by running the following commands:
```
cd src/perception/perception
sudo python3 setup.py install
```
Then, you can test this package on your webcam:
```
ros2 run perception cam_publisher  --ros-args -p source:='0'  # to get images from streaming, specify the camera
ros2 run perception detect_objects --ros-args -p visualize:=0
ros2 run perception bbox_subscriber # to get bounding boxes
```

### Localization
The ```localization``` package is used to estimate robot's states from different sensors (IMU, GPS).

To work on the localization part, run the following instructions:
```
ros2 run localization gps_publisher  # to get GPS data
ros2 run localization acc_publisher  # to get Acc data
ros2 run localization gyro_publisher # to get Gyro data
ros2 run localization localize_robot # taking all sensors data and run EKF to estimate states (position, velocity and orientation)
```

### Brain
The ```brain``` package is the main one. When sensors are ON (Camera, GPS, IMU), thsi package is used to collect all publlished informations (states + Bounding boxes) to be used, right after, in motion planning and control.

If the other packages work well, you can then run the following command:
```
ros2 run brain brain_subscriber
```
