from setuptools import find_packages, setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='yoso-mmdet',
    version='0.1',
    packages=find_packages(include=['yoso', 'yoso.*']),
    install_requires=parse_requirements('requirements.txt'),
    author='romiptr',
    description='YOSO implementation on MMDetection',
)