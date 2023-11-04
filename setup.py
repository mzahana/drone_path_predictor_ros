from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'DronePathPredictor_ros'

# Get all the launch files
launch_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'launch', '*.launch.py'))]

# Get all the config files
config_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'config', '*'))]

# Get all the test files
test_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'test', '*.py'))]

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # Removed the exclude argument
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
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
            'trajectory_predictor_node = DronePathPredictor_ros.trajectory_predictor_node:main',
        ],
    },
)

