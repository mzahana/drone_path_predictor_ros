from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'drone_path_predictor_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohamed Abdelkader',
    maintainer_email='mohamedashraf123@gmail.com',
    description='A ROS 2 package for predicting drone paths',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_predictor_node = drone_path_predictor_ros.trajectory_predictor_node:main',
        ],
    },
)
