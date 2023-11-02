from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'DronePathPredictor_ros'

# Get all the launch files
launch_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'launch', '*.launch.py'))]

# Get all the config files
config_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'config', '*'))]

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all files in launch and config directories
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'config'), config_files),
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
            #'predictor = drone_path_predictor_ros.predictor:main',
        ],
    },
)
