from setuptools import setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alisahili',
    maintainer_email='ali@gerard.farm',
    description='Localization Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_publisher = localization.gps_pub:main',
            'acc_publisher = localization.acc_pub:main',
            'gyro_publisher = localization.gyro_pub:main',
            'localize_robot = localization.localize_sub_pub:main',
        ],
    },
)
