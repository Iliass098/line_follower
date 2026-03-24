import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('line_follower_jetson')
    params_file = os.path.join(package_share, 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package='line_follower_jetson',
            executable='line_follower_node',
            name='line_follower_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
