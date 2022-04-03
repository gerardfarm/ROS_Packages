# Import the necessary libraries
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes

from custom_msgs.msg._my_sensor_msg import MySensorMsg  # noqa: F401

import numpy as np

class Acc_Publisher(Node):
  """
  Create a Acc_Publisher class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('acc_publisher')
      
    # Create the publisher. This publisher will publish Acc data
    self.publisher_ = self.create_publisher(MySensorMsg, 'acc_data', 10)
    timer_period = 0.1  # We will publish a message every 0.1 seconds
    self.timer = self.create_timer(timer_period, self.timer_callback)

  def timer_callback(self):
    """
    Callback function.
    This function gets called every 0.1 seconds.
    """
    msg = MySensorMsg()
    msg.t = 0.01
    msg.data = np.array([0., 0., 0.])

    self.publisher_.publish(msg)
 
    # Display the message on the console
    self.get_logger().info('Publishing Acc data')
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  acc_publisher = Acc_Publisher()
  
  # Spin the node so the callback function is called.
  rclpy.spin(acc_publisher)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  acc_publisher.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
