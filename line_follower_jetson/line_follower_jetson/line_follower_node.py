#!/usr/bin/env python3
"""ROS 2 Foxy line follower node for Jetson-based mobile robots."""

from collections import deque
import math

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image


class LineFollowerNode(Node):
    """Line following node using OpenCV and a discrete PID controller."""

    def __init__(self) -> None:
        super().__init__('line_follower_node')

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_stamp = None
        self.last_processed_stamp_ns = None

        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_control_time = None
        self.last_line_center_x = None
        self.line_loss_count = 0
        self.recent_centers = deque(maxlen=5)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/camera/image_raw'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('debug_image_topic', '/camera/line_debug'),
                ('processing_rate_hz', 15.0),
                ('linear_speed', 0.12),
                ('reduced_linear_speed', 0.05),
                ('max_angular_speed', 1.5),
                ('Kp', 0.0045),
                ('Ki', 0.0002),
                ('Kd', 0.0020),
                ('integral_limit', 3000.0),
                ('line_color', 'black'),
                ('use_canny', False),
                ('use_hough_assist', True),
                ('hough_center_band_ratio', 0.55),
                ('roi_vertical_start', 0.60),
                ('roi_vertical_end', 1.00),
                ('blur_kernel_size', 5),
                ('threshold_value', 120),
                ('adaptive_threshold', True),
                ('canny_low_threshold', 50),
                ('canny_high_threshold', 150),
                ('min_contour_area', 700.0),
                ('line_loss_limit', 6),
                ('image_timeout_sec', 0.5),
                ('ambiguity_ratio_threshold', 0.35),
                ('max_center_jump_px', 140.0),
                ('publish_debug_image', True),
                ('show_debug_window', False),
                ('debug_log_interval_sec', 1.0),
            ],
        )

        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10,
        )

        timer_period = 1.0 / max(self.processing_rate_hz, 1.0)
        self.processing_timer = self.create_timer(timer_period, self.process_latest_image)

        self.last_debug_log_time = self.get_clock().now()
        self.get_logger().info(
            'Line follower node started. camera_topic=%s, cmd_vel_topic=%s, rate=%.2f Hz'
            % (self.camera_topic, self.cmd_vel_topic, self.processing_rate_hz)
        )

    def _load_parameters(self) -> None:
        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.processing_rate_hz = float(self.get_parameter('processing_rate_hz').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.reduced_linear_speed = float(self.get_parameter('reduced_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.kp = float(self.get_parameter('Kp').value)
        self.ki = float(self.get_parameter('Ki').value)
        self.kd = float(self.get_parameter('Kd').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)
        self.line_color = str(self.get_parameter('line_color').value).lower().strip()
        self.use_canny = bool(self.get_parameter('use_canny').value)
        self.use_hough_assist = bool(self.get_parameter('use_hough_assist').value)
        self.hough_center_band_ratio = float(self.get_parameter('hough_center_band_ratio').value)
        self.roi_vertical_start = float(self.get_parameter('roi_vertical_start').value)
        self.roi_vertical_end = float(self.get_parameter('roi_vertical_end').value)
        self.blur_kernel_size = int(self.get_parameter('blur_kernel_size').value)
        self.threshold_value = int(self.get_parameter('threshold_value').value)
        self.adaptive_threshold = bool(self.get_parameter('adaptive_threshold').value)
        self.canny_low_threshold = int(self.get_parameter('canny_low_threshold').value)
        self.canny_high_threshold = int(self.get_parameter('canny_high_threshold').value)
        self.min_contour_area = float(self.get_parameter('min_contour_area').value)
        self.line_loss_limit = int(self.get_parameter('line_loss_limit').value)
        self.image_timeout_sec = float(self.get_parameter('image_timeout_sec').value)
        self.ambiguity_ratio_threshold = float(self.get_parameter('ambiguity_ratio_threshold').value)
        self.max_center_jump_px = float(self.get_parameter('max_center_jump_px').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)
        self.debug_log_interval_sec = float(self.get_parameter('debug_log_interval_sec').value)

        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1

        self.roi_vertical_start = min(max(self.roi_vertical_start, 0.0), 0.95)
        self.roi_vertical_end = min(max(self.roi_vertical_end, self.roi_vertical_start + 0.05), 1.0)
        self.hough_center_band_ratio = min(max(self.hough_center_band_ratio, 0.15), 1.0)

    def _on_parameters_changed(self, params):
        for param in params:
            if param.name in {'Kp', 'Ki', 'Kd'} and param.value < 0.0:
                return SetParametersResult(successful=False, reason='PID gains must be >= 0')
            if param.name == 'processing_rate_hz' and param.value <= 0.0:
                return SetParametersResult(successful=False, reason='processing_rate_hz must be > 0')
            if param.name == 'blur_kernel_size' and int(param.value) < 1:
                return SetParametersResult(successful=False, reason='blur_kernel_size must be >= 1')
            if param.name == 'line_color' and str(param.value).lower() not in {'black', 'white'}:
                return SetParametersResult(successful=False, reason='line_color must be black or white')
            if param.name == 'hough_center_band_ratio' and not (0.0 < float(param.value) <= 1.0):
                return SetParametersResult(successful=False, reason='hough_center_band_ratio must be in (0, 1]')

        self._load_parameters()
        new_period = 1.0 / max(self.processing_rate_hz, 1.0)
        if hasattr(self, 'processing_timer') and self.processing_timer is not None:
            self.destroy_timer(self.processing_timer)
        self.processing_timer = self.create_timer(new_period, self.process_latest_image)
        self.get_logger().info('Parameters updated at runtime.')
        return SetParametersResult(successful=True)

    def image_callback(self, msg: Image) -> None:
        self.latest_image = msg
        self.latest_stamp = msg.header.stamp

    def process_latest_image(self) -> None:
        if self.latest_image is None:
            self._publish_stop_if_stale('No images received yet.')
            return

        stamp_ns = (
            int(self.latest_image.header.stamp.sec) * 1_000_000_000
            + int(self.latest_image.header.stamp.nanosec)
        )
        if self.last_processed_stamp_ns == stamp_ns:
            return

        if self._is_image_stale(self.latest_image.header.stamp):
            self._publish_stop_if_stale('Camera image timeout.')
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as exc:  # pragma: no cover
            self.get_logger().error('cv_bridge conversion failed: %s' % str(exc))
            self.publish_stop()
            return

        self.last_processed_stamp_ns = stamp_ns
        self._process_frame(frame, self.latest_image.header.stamp)

    def _is_image_stale(self, stamp) -> bool:
        msg_time = Time.from_msg(stamp)
        age = self.get_clock().now() - msg_time
        return age > Duration(seconds=self.image_timeout_sec)

    def _process_frame(self, frame: np.ndarray, stamp) -> None:
        height, width = frame.shape[:2]
        roi_top = int(height * self.roi_vertical_start)
        roi_bottom = int(height * self.roi_vertical_end)
        roi = frame[roi_top:roi_bottom, :]

        if roi.size == 0:
            self.get_logger().warning('Empty ROI, stopping robot.')
            self.publish_stop()
            return

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        if self.adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
        else:
            _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)

        if self.line_color == 'black':
            binary = cv2.bitwise_not(binary)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        edges = cv2.Canny(binary, self.canny_low_threshold, self.canny_high_threshold)

        detection = self._detect_line(binary, edges)
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (0, roi_top), (width - 1, roi_bottom - 1), (255, 255, 0), 2)
        image_center_x = width * 0.5

        if detection is None:
            self.line_loss_count += 1
            self._handle_line_loss(debug_frame, roi_top, width, stamp)
            return

        line_center_x, confidence, method_name, contour_area = detection
        line_center_x = self._smooth_center(line_center_x)
        error = image_center_x - float(line_center_x)

        control_now = self.get_clock().now()
        dt = self._compute_dt(control_now)
        angular_command = self._compute_pid(error, dt)
        angular_command = float(np.clip(angular_command, -self.max_angular_speed, self.max_angular_speed))

        linear_command = self.linear_speed
        if confidence < 0.55:
            linear_command = min(self.linear_speed, self.reduced_linear_speed)

        twist = Twist()
        twist.linear.x = float(linear_command)
        twist.angular.z = angular_command
        self.cmd_pub.publish(twist)

        self.line_loss_count = 0
        self.last_line_center_x = line_center_x

        self._draw_debug(
            debug_frame,
            binary,
            roi_top,
            line_center_x,
            image_center_x,
            method_name,
            error,
            angular_command,
            contour_area,
            confidence,
        )
        self._publish_debug(debug_frame, stamp)
        self._log_status(error, angular_command, confidence, method_name)

    def _detect_line(self, binary: np.ndarray, edges: np.ndarray):
        hough_result = self._detect_with_hough(binary, edges)
        if hough_result is not None:
            return hough_result

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best_contour = contours[0]
        best_area = cv2.contourArea(best_contour)
        if best_area < self.min_contour_area:
            return None

        moments = cv2.moments(best_contour)
        if moments['m00'] <= 0.0:
            return None

        cx = moments['m10'] / moments['m00']
        confidence = float(np.clip(best_area / max(self.min_contour_area * 4.0, 1.0), 0.0, 0.45))
        return float(cx), confidence, 'centroid_fallback', best_area

    def _detect_with_hough(self, binary: np.ndarray, edges: np.ndarray):
        if not self.use_hough_assist:
            return None

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=max(20, int(binary.shape[1] * 0.10)),
            maxLineGap=25,
        )
        if lines is None:
            return None

        image_height, image_width = binary.shape[:2]
        image_center_x = image_width * 0.5
        band_half_width = max(1.0, 0.5 * image_width * self.hough_center_band_ratio)
        band_left = image_center_x - band_half_width
        band_right = image_center_x + band_half_width

        preferred_centers = []
        preferred_weights = []
        fallback_centers = []
        fallback_weights = []

        for segment in lines[:, 0]:
            x1, y1, x2, y2 = [float(v) for v in segment]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 10.0:
                continue

            if y1 >= y2:
                lower_x = x1
                lower_y = y1
            else:
                lower_x = x2
                lower_y = y2

            center_x = 0.5 * (x1 + x2)
            bottom_score = float(np.clip(lower_y / max(image_height - 1.0, 1.0), 0.0, 1.0))
            center_distance = abs(lower_x - image_center_x) / max(image_center_x, 1.0)
            center_score = float(np.clip(1.0 - center_distance, 0.0, 1.0))
            weight = length * (0.65 + 0.35 * bottom_score) * (0.40 + 0.60 * center_score)

            fallback_centers.append(center_x)
            fallback_weights.append(weight)

            if band_left <= lower_x <= band_right:
                preferred_centers.append(center_x)
                preferred_weights.append(weight)

        weights = preferred_weights if preferred_weights else fallback_weights
        centers = preferred_centers if preferred_centers else fallback_centers
        if not weights:
            return None

        cx = float(np.average(centers, weights=weights))
        confidence = float(np.clip(sum(weights) / max(image_width * 2.0, 1.0), 0.0, 0.85))
        if not preferred_weights:
            confidence *= 0.6

        if self.last_line_center_x is not None and abs(cx - self.last_line_center_x) > self.max_center_jump_px:
            confidence *= 0.7

        return cx, confidence, 'hough_front_center', float(sum(weights))

    def _smooth_center(self, center_x: float) -> float:
        self.recent_centers.append(center_x)
        return float(sum(self.recent_centers) / len(self.recent_centers))

    def _compute_dt(self, now) -> float:
        if self.last_control_time is None:
            self.last_control_time = now
            return 1.0 / max(self.processing_rate_hz, 1.0)

        dt_ns = (now - self.last_control_time).nanoseconds
        self.last_control_time = now
        dt = max(dt_ns / 1e9, 1e-3)
        return dt

    def _compute_pid(self, error: float, dt: float) -> float:
        self.integral_error += error * dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)

    def _handle_line_loss(self, debug_frame, roi_top: int, width: int, stamp) -> None:
        twist = Twist()
        if self.line_loss_count < self.line_loss_limit and self.last_line_center_x is not None:
            twist.linear.x = float(self.reduced_linear_speed)
            twist.angular.z = float(
                np.clip(
                    self.kp * (width * 0.5 - self.last_line_center_x),
                    -0.5 * self.max_angular_speed,
                    0.5 * self.max_angular_speed,
                )
            )
            self.cmd_pub.publish(twist)
        else:
            self.publish_stop(reset_pid=True)

        cv2.putText(
            debug_frame,
            'Line lost: slowing/stopping',
            (20, max(30, roi_top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        self._publish_debug(debug_frame, stamp)
        self._log_status(float('nan'), twist.angular.z, 0.0, 'line_lost')

    def _draw_debug(
        self,
        debug_frame: np.ndarray,
        binary: np.ndarray,
        roi_top: int,
        line_center_x: float,
        image_center_x: float,
        method_name: str,
        error: float,
        angular_command: float,
        contour_area: float,
        confidence: float,
    ) -> None:
        roi_height = binary.shape[0]
        line_x_int = int(round(line_center_x))
        image_x_int = int(round(image_center_x))
        band_half_width = max(1.0, 0.5 * debug_frame.shape[1] * self.hough_center_band_ratio)
        band_left = int(round(image_center_x - band_half_width))
        band_right = int(round(image_center_x + band_half_width))

        cv2.rectangle(
            debug_frame,
            (band_left, roi_top),
            (band_right, roi_top + roi_height),
            (255, 0, 255),
            1,
        )
        cv2.line(
            debug_frame,
            (image_x_int, roi_top),
            (image_x_int, roi_top + roi_height),
            (0, 255, 255),
            2,
        )
        cv2.line(
            debug_frame,
            (line_x_int, roi_top),
            (line_x_int, roi_top + roi_height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            'method=%s err=%.1f ang=%.3f conf=%.2f area=%.1f'
            % (method_name, error, angular_command, confidence, contour_area),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _publish_debug(self, debug_frame: np.ndarray, stamp) -> None:
        if self.publish_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header.stamp = stamp
            self.debug_pub.publish(debug_msg)

        if self.show_debug_window:
            cv2.imshow('line_follower_debug', debug_frame)
            cv2.waitKey(1)

    def _log_status(
        self,
        error: float,
        angular_command: float,
        confidence: float,
        method_name: str,
    ) -> None:
        now = self.get_clock().now()
        if (now - self.last_debug_log_time).nanoseconds < int(self.debug_log_interval_sec * 1e9):
            return

        self.last_debug_log_time = now
        if math.isnan(error):
            self.get_logger().warn(
                'Line not detected reliably. method=%s angular=%.3f loss_count=%d'
                % (method_name, angular_command, self.line_loss_count)
            )
            return

        self.get_logger().info(
            'Tracking line. method=%s error=%.2f angular=%.3f confidence=%.2f'
            % (method_name, error, angular_command, confidence)
        )

    def _publish_stop_if_stale(self, reason: str) -> None:
        self.publish_stop()
        self.get_logger().debug(reason)

    def publish_stop(self, reset_pid: bool = False) -> None:
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        if reset_pid:
            self.integral_error = 0.0
            self.previous_error = 0.0
            self.recent_centers.clear()
            self.last_line_center_x = None
            self.line_loss_count = 0

    def destroy_node(self):
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop(reset_pid=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
