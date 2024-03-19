# from setuptools import setup, find_packages
# import os
# from glob import glob

# package_name = 'DronePathPredictor_ros'

# # Get all the launch files
# launch_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'launch', '*.launch.py'))]

# # Get all the config files
# config_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'config', '*'))]

# # Get all the test files
# test_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'test', '*.py'))]

# setup(
#     name=package_name,
#     version='0.0.1',
#     packages=find_packages(),  # Removed the exclude argument
#     data_files=[
#         (os.path.join('share', package_name), ['package.xml']),
#         (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
#         (os.path.join('share', package_name, 'config'), glob('config/*.*')),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='Mohamed Abdelkader',
#     maintainer_email='mohamedashraf123@gmail.com',
#     description='A ROS 2 package for predicting drone paths',
#     license='MIT',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'trajectory_predictor_node = DronePathPredictor_ros.trajectory_predictor_node:main',
#         ],
#     },
# )

from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'drone_path_predictor_ros'

# Function to recursively list all files in a directory
def recursive_glob(root_dir, file_pattern):
    return [os.path.relpath(os.path.join(dirpath, file), package_name)
            for dirpath, dirnames, files in os.walk(root_dir)
            for file in files if file.endswith(file_pattern)]

# Get all the launch files
launch_files = recursive_glob(os.path.join(package_name, 'launch'), '*.launch.py')

# # Get all the config files and subdirectory files
# config_files = recursive_glob(os.path.join(package_name, 'config'), ['*.*', '*.yaml', '*.yml'])
# Get all the config files
config_files = [os.path.relpath(f, package_name) for f in glob(os.path.join(package_name, 'config', '*'))]

# Get all the test files
test_files = recursive_glob(os.path.join(package_name, 'test'), '*.py')

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        # Add any other directories you need to include
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
