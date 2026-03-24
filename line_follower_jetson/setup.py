from glob import glob
import os

from setuptools import setup


package_name = 'line_follower_jetson'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ilias',
    maintainer_email='iliasselkamilili@gmail.com',
    description='ROS 2 Foxy line follower for Jetson using OpenCV and PID control.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_follower_node = line_follower_jetson.line_follower_node:main',
        ],
    },
)

