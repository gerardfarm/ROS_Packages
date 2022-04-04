from setuptools import setup

package_name = 'perception'
submodules = 'perception/object_detection'
sub_submodules1 = 'perception/object_detection/helpers'
sub_submodules2 = 'perception/object_detection/models'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules, 
                sub_submodules1, sub_submodules2],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alisahili',
    maintainer_email='ali@gerard.farm',
    description='Perception Package for autonomous navigation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam_publisher = perception.cam_pub:main',
            'cam_subscriber = perception.cam_sub:main',
            'detect_objects = perception.detect_sub_pub:main',
            'bbox_subscriber = perception.bbox_sub:main',
        ],
    },
)
