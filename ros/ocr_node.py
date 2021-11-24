
"""
The MIT License (MIT)
Copyright (c) 2021 NVIDIA CORPORATION
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

class OcrNode(Node):
    def __init__(self):
        super().__init__("OCR_Node")

        self._text_pub = self.create_publisher(String, 'ocr_text', 10)
        self._timer = self.create_timer(0.1, self.callback) # 10HZ timer
        self._bridge = CvBridge()

        print("Loading model...")
        self._reader = easyocr.Reader(["en"], use_trt=False)

    def callback(self):
        arr = cv2.imread("messi.jpg")

        msg = String()
        result = reader.readtext(frame, text_threshold=0.85)
        msg.data = "hello"
        self._text_pub.publish(msg)
        self.get_logger().info(f"Publishing string")

if __name__ == "__main__":
    rclpy.init()

    node = OcrNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()