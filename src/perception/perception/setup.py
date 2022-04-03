from setuptools import setup, find_packages

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

requirements = read_requirements("requirements.txt")

setup(
    name = 'object_detection',
    version = '0.1.0',
    author = 'Ali Rida Sahili',
    author_email = 'ali@gerard.farm',
    url = '',
    description = 'An Object Detection python package.',
    license = "MIT license",
    packages = ['object_detection', 'object_detection/helpers'],
    install_requires = requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]  # Update these accordingly
)
