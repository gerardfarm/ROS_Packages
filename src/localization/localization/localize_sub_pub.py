import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from custom_msgs.msg._my_sensor_msg import MySensorMsg  # noqa: F401
from custom_msgs.msg._states_msg import StatesMsg

import numpy as np
from localization.es_ekf import EKF


class Localize_sub_pub(Node):

    def __init__(self):
        super().__init__('localize_subscriber_publisher')
        
        # print(get_package_share_directory('cv_basics'));assert(0)

        # EKF object
        self.ekf = EKF()
        self.start = True

        # Create the publisher. This publisher will send infos about states
        self.publisher_ = self.create_publisher(StatesMsg, 'estimated_states', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create subscribers. These subscribers will receive sensors data
        self.subscription_gps = self.create_subscription(
                MySensorMsg, # TODO
                'gps_data', self.gps_callback, 10)
        
        self.subscription_acc = self.create_subscription(
                MySensorMsg, # TODO
                'acc_data', self.acc_callback, 10) 

        self.subscription_gyro = self.create_subscription(
                MySensorMsg, # TODO
                'gyro_data', self.gyro_callback, 10) 

        self.subscription_gps   # prevent unused variable warning
        self.subscription_acc   # prevent unused variable warning
        self.subscription_gyro  # prevent unused variable warning

        self.gps_check = False
        self.acc_check = False
        self.gyro_check = False

    def gps_callback(self, data):
        """ Callback Function  """
        # Display the message on the console
        self.get_logger().info('Receiving GPS data')
        self.gps_data = data
        self.gps_check = True

    def acc_callback(self, data):
        """ Callback Function  """
        # Display the message on the console
        self.get_logger().info('Receiving Acc data')
        self.acc_data = data
        self.acc_check = True

    def gyro_callback(self, data):
        """ Callback Function  """
        # Display the message on the console
        self.get_logger().info('Receiving Gyro data')
        self.gyro_data = data
        self.gyro_check = True

    def timer_callback(self):
        
        if self.gps_check and self.acc_check and self.gyro_check:
            p_est, v_est, \
              q_est, \
                p_cov = self.ekf.Predict_and_Update(self.acc_data.t,
                                                    self.gps_data.t,
                                                    self.gps_data.data,
                                                    self.acc_data.t,
                                                    self.acc_data.data,
                                                    self.gyro_data.data,
                                                    start=self.start)
            self.start = False
            states = StatesMsg()
            states.p = np.array(p_est, dtype=np.float32)
            states.v = np.array(v_est, dtype=np.float32)
            states.q = np.array(q_est, dtype=np.float32)

            self.publisher_.publish(states)
            self.get_logger().info('Publishing estimated states [positions, velocities, orientations]')

            self.gps_check = False
            self.acc_check = False
            self.gyro_check = False
        else:
            if not self.gps_check:
                self.get_logger().info('No GPS signal!!')
            if not self.acc_check:
                self.get_logger().info('No ACCelerometer signal!!')
            if not self.gyro_check:
                self.get_logger().info('No Gyroscope signal!!')

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
  
    # Create the node
    localize_sub_and_pub = Localize_sub_pub()
    
    # Spin the node so the callback function is called.
    rclpy.spin(localize_sub_and_pub)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    localize_sub_and_pub.destroy_node()
    
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()

