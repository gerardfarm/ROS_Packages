# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from std_msgs.msg import String
from custom_msgs.msg._b_box_list import BBoxList  # noqa: F401

import numpy as np
 
class BBox_Subscriber(Node):
  """
  Create an BBox_Subscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('bbox_subscriber')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      BBoxList, 
      'Bounding_boxes', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
         
  def listener_callback(self, data):
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
      print(coords, score, cls)


def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  bbox_subscriber = BBox_Subscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(bbox_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  bbox_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
