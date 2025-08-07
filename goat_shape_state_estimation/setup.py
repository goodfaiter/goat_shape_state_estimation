from setuptools import setup

package_name = 'goat_shape_state_estimation'

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
    maintainer='Valentin Yuryev',
    maintainer_email='valentin.yuryev@epfl.ch',
    description='The GOAT shape and state estimation package.',
    license='MIT',
    entry_points={
        'console_scripts': [
                'goat_shape_state_estimation = goat_shape_state_estimation.goat_shape_state_estimation:main',
        ],
    },
)