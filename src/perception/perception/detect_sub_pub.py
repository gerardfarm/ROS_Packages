
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from std_msgs.msg import String
from custom_msgs.msg._b_box import BBox  # noqa: F401
from custom_msgs.msg._b_box_list import BBoxList  # noqa: F401

import time
import logging

import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
import torch.backends.cudnn as cudnn

from .object_detection.helpers.utils import check_img_size, time_synchronized
from .object_detection.helpers.boxes import non_max_suppression, scale_coords
from .object_detection.helpers.classifier import load_classifier, apply_classifier
from .object_detection.helpers.plots import colors, plot_one_box
from .object_detection.models.experimental import attempt_load

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        
        # print(get_package_share_directory('cv_basics'));assert(0)
        
        weights='/home/alisahili/Gerard_Farm/ros_ws/src/perception/perception/yolov5s.pth'  # model.pt path(s)
        cfg = '/home/alisahili/Gerard_Farm/ros_ws/src/perception/perception/object_detection/models/yolov5s.yaml'
        hyp = '/home/alisahili/Gerard_Farm/ros_ws/src/perception/perception/hyp.scratch.yaml'

        self.source = '0'    # 0 for webcam
        self.view_img=True  # show results

        self.imgsz=640  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.stride = 32

        # Initialize
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device, cfg=cfg, hyp=hyp)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet50', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()

        # Dataloader
        # view_img = check_imshow() # True
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Create the publisher. This publisher will send infos about detected objects
        self.publisher_ = self.create_publisher(BBoxList, 'Bounding_boxes', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create the subscriber. This subscriber will receive an Image
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.camera_callback,
            10) # 10 is the number of frames per seconds
        self.subscription  # prevent unused variable warning

        self.publish_ok = False

    def camera_callback(self, data):
        """ Callback Function  """
        # Display the message on the console
        self.get_logger().info('Receiving video frame')

        t0 = time.time()

        # Convert ROS Image message to OpenCV image
        img = self.br.imgmsg_to_cv2(data) # Current frame
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation = cv2.INTER_AREA)

        # Letterbox
        img0 = img.copy()
        img =  img[np.newaxis, :, :, :] # [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        self.pred = self.model(img,
                     augment=self.augment,
                     visualize=False)[0]

        self.img = img
        self.img0 = img0

        # Apply NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, 
                            self.classes, self.agnostic_nms, max_det=self.max_det)
                            
        t2 = time_synchronized()

        # Apply Classifier
        if self.classify:
            self.pred = apply_classifier(self.pred, self.modelc, img, img0)


        # Process detections
        for i, det in enumerate(self.pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=self.line_thickness)

            # Print time (inference + NMS)
            self.get_logger().info(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if self.view_img:
                cv2.imshow("IMAGE", img0)
                cv2.waitKey(1)  # 1 millisecond

    def timer_callback(self):
        
        boxes_msg = BBoxList()

        # Process detections
        for i, det in enumerate(self.pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % self.img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], self.img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')

                    obj = BBox()
                    obj.bbox_coords = np.array([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])], dtype=np.uint32)
                    obj.class_label = int(cls)
                    obj.score = float(conf)

                    boxes_msg.bboxes.append(obj)

        self.publisher_.publish(boxes_msg)
        self.get_logger().info('Publishing bounding boxes')


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
  
    # Create the node
    camera_subscriber = Camera_subscriber()
    
    # Spin the node so the callback function is called.
    rclpy.spin(camera_subscriber)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_subscriber.destroy_node()
    
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()

