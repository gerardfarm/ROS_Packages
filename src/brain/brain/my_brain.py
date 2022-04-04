# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node

from custom_msgs.msg._b_box_list import BBoxList  # noqa: F401
from custom_msgs.msg._states_msg import StatesMsg

import numpy as np
 
class Brain_Subscriber(Node):
  """
  Create an BBox_Subscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('brain_subscriber')
      
    # Create the bboxes subscriber. The queue size is 10 messages.
    self.subscription_boxes = self.create_subscription(
      BBoxList, 
      'Bounding_boxes', 
      self.listenerBBoxes_callback, 
      10)

    # Create the States subscriber. The queue size is 10 messages.
    self.subscription_states = self.create_subscription(
      StatesMsg, 
      'estimated_states', 
      self.listenerStates_callback, 
      10)

    self.subscription_states # prevent unused variable warning
         
  def listenerBBoxes_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving bounding boxes data:')
    bboxes_list = data.bboxes

    for bbox in bboxes_list:
      coords = bbox.bbox_coords
      score = bbox.score
      cls = bbox.class_label
      print("Boxes: ", coords, score, cls)

  def listenerStates_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving Estimated states:')
    position, velocity, quaternion = data.p, data.v, data.q
    print("States: ", position, velocity, quaternion)


def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  brain_subscriber = Brain_Subscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(brain_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  brain_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
