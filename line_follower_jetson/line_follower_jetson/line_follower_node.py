#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np


class LineFollower(Node):

    def __init__(self):
        super().__init__('line_follower_simple')

        self.bridge = CvBridge()

        # Subs y pubs
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parámetros simples
        self.linear_speed = 0.1
        self.kp = 0.005

        self.get_logger().info("Line follower simple iniciado")

    def image_callback(self, msg):
        # Convertir imagen
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        height, width = frame.shape[:2]

        # ROI (solo parte inferior)
        roi = frame[int(height * 0.6):height, :]

        # ===== DETECCIÓN SIMPLE DE LÍNEA NEGRA =====

        # Escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold invertido → negro = blanco
        _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

        # Limpiar ruido
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # ===========================================

        # Buscar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()

        if contours:
            # Contorno más grande
            c = max(contours, key=cv2.contourArea)

            # Ignorar ruido pequeño
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    # Centro imagen
                    error = (width / 2) - cx

                    # Control proporcional
                    twist.linear.x = self.linear_speed
                    twist.angular.z = self.kp * error

                    # Debug visual
                    cv2.circle(roi, (cx, 50), 10, (0, 255, 0), -1)
                    cv2.line(roi, (int(width/2), 0), (int(width/2), 100), (255, 0, 0), 2)

                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        else:
            # Línea perdida
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # Publicar comando
        self.pub.publish(twist)

        # Mostrar debug (opcional)
        cv2.imshow("binary", binary)
        cv2.imshow("roi", roi)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
