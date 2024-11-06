#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self):
        self.prev_gray = None
        self.transforms = []

    def stabilize_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 最初のフレームを基準として保存
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        # 前のフレームと現在のフレーム間の動きを推定
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 平均動きを算出し、平滑化
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        self.transforms.append((dx, dy))

        # 最近の5フレームの動きを平滑化
        if len(self.transforms) > 5:
            smoothed_dx = np.mean([t[0] for t in self.transforms[-5:]])
            smoothed_dy = np.mean([t[1] for t in self.transforms[-5:]])
        else:
            smoothed_dx = dx
            smoothed_dy = dy

        # 平滑化した移動量でフレームを補正
        transform_matrix = np.float32([[1, 0, -smoothed_dx], [0, 1, -smoothed_dy]])
        stabilized_frame = cv2.warpAffine(frame, transform_matrix, (frame.shape[1], frame.shape[0]))

        # 現フレームを次回の基準として保存
        self.prev_gray = gray

        return stabilized_frame

class StabilizedImagePublisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.stabilizer = VideoStabilizer()
        # Subscribe to the original compressed image topic
        self.image_sub = rospy.Subscriber("/image/mercator/compressed", CompressedImage, self.image_callback)
        # Publish to the new compressed image topic
        self.compressed_image_pub = rospy.Publisher("/image/mercator/compressed2", CompressedImage, queue_size=10)

    def image_callback(self, msg):
        try:
            # Convert CompressedImage to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            # Stabilize the frame
            stabilized_frame = self.stabilizer.stabilize_frame(cv_image)

            # Convert back to compressed format
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = self.bridge.cv2_to_compressed_imgmsg(stabilized_frame).data

            # Publish to the new topic
            self.compressed_image_pub.publish(compressed_msg)

        except Exception as e:
            rospy.logerr("Image stabilization failed: %s" % e)

def main():
    rospy.init_node('stabilized_image_publisher', anonymous=True)
    stabilizer = StabilizedImagePublisher()
    rospy.spin()

if __name__ == '__main__':
    main()

