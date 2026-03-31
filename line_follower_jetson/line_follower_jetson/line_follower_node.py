#!/usr/bin/env python3
"""
Siguelíneas para JetBot con ROS2
----------------------------------
- ROI: franja inferior de la imagen (configurable)
- Espacio de color: HSV para detección robusta de la línea
- Control: PID sobre el error lateral del centroide
- Tópicos:
    Suscripción : /robot6/camera/image_raw  (sensor_msgs/Image)
    Publicación : /robot6/cmd_vel           (geometry_msgs/Twist)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


# ──────────────────────────────────────────────
#  Parámetros HSV  (línea NEGRA sobre fondo claro)
#  Ajusta estos rangos según tu pista
# ──────────────────────────────────────────────
HSV_LOWER = np.array([0,   0,   0  ])   # negro
HSV_UPPER = np.array([180, 255, 40 ])   # negro

# Para línea BLANCA descomenta lo siguiente y comenta lo de arriba:
# HSV_LOWER = np.array([0,   0,   200])
# HSV_UPPER = np.array([180, 30,  255])

# Para línea AMARILLA:
# HSV_LOWER = np.array([20,  100, 100])
# HSV_UPPER = np.array([35,  255, 255])


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__('line_follower')

        # ── Parámetros ROS2 (modificables en tiempo de ejecución) ──
        self.declare_parameter('roi_top_fraction', 0.20)   # fracción de imagen que se ignora por arriba
        self.declare_parameter('base_speed',       0.10 )   # m/s velocidad lineal base
        self.declare_parameter('max_angular',      2.5)    # rad/s giro máximo
        self.declare_parameter('kp',               0.003)  # ganancia proporcional
        self.declare_parameter('ki',               0.0) # ganancia integral
        self.declare_parameter('kd',               0.0)  # ganancia derivativa
        self.declare_parameter('lost_stop',        True)   # para si pierde la línea

        # ── Puente OpenCV ──
        self.bridge = CvBridge()

        # ── Estado PID ──
        self._prev_error  = 0.0
        self._integral    = 0.0
        self._integral_max = 300.0   # anti-windup
        self._last_stamp  = None

        # ── Publicador / Suscriptor ──
        self.pub_cmd = self.create_publisher(Twist, '/robot6/cmd_vel', 10)
        self.sub_img = self.create_subscription(
            Image,
            '/robot6/camera/image_raw',
            self.image_callback,
            10
        )
        self.pub_debug = self.create_publisher(Image, '/robot6/debug/mask', 10)
        self.get_logger().info('✅  Nodo siguelíneas iniciado')

    # ─────────────────────────────────────────
    def image_callback(self, msg: Image):
        # 1. Convertir ROS Image → OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w  = frame.shape[:2]

        # 2. ROI: sólo la parte inferior
        roi_fraction = self.get_parameter('roi_top_fraction').value
        roi_y = int(h * roi_fraction)
        roi   = frame[roi_y:, :]          # de roi_y hasta el final

        # 3. HSV + máscara
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Morfología para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        # 4. Centroide de la línea
        cx = self._get_centroid(mask)

        # 5. PID
        cmd = Twist()
        if cx is None:
            # Línea perdida
            if self.get_parameter('lost_stop').value:
                self.pub_cmd.publish(cmd)   # stop
            self.get_logger().warn('⚠️  Línea no detectada')
            self._integral = 0.0
            return

        # Error: desviación respecto al centro horizontal del ROI
        roi_w   = roi.shape[1]
        error   = float(cx - roi_w // 2)   # negativo → izquierda, positivo → derecha

        # dt
        now = self.get_clock().now()
        if self._last_stamp is None:
            dt = 0.033
        else:
            dt = (now - self._last_stamp).nanoseconds * 1e-9
            dt = max(dt, 1e-4)
        self._last_stamp = now

        # PID
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value

        self._integral  += error * dt
        self._integral   = np.clip(self._integral,
                                   -self._integral_max,
                                    self._integral_max)
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error

        angular = -(kp * error + ki * self._integral + kd * derivative)
        angular  = np.clip(angular,
                           -self.get_parameter('max_angular').value,
                            self.get_parameter('max_angular').value)

        # Velocidad lineal: reduce al girar
        base_speed = self.get_parameter('base_speed').value
        linear     = base_speed * (1.0 - 0.5 * abs(angular) /
                                   self.get_parameter('max_angular').value)

        cmd.linear.x  = linear
        cmd.angular.z = angular
        self.pub_cmd.publish(cmd)

        self.get_logger().debug(
            f'cx={cx:4d}  err={error:+.1f}  ang={angular:+.3f}'
        )

    # ─────────────────────────────────────────
    @staticmethod
    def _get_centroid(mask: np.ndarray):
        """Devuelve la coordenada X del centroide de la máscara o None."""
        M = cv2.moments(mask)
        if M['m00'] < 500:   # área mínima para evitar falsos positivos
            return None
        return int(M['m10'] / M['m00'])


# ─────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parar el robot al salir
        stop = Twist()
        node.pub_cmd.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
