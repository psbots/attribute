from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='attribute',
    version='0.0.1',
    description='Neural Network Interpretability and Visualisation library for TensorFlow 2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    keywords='saliency attribution mask neural network deep learning',
    packages=['attribute'],
    install_requires=['numpy', 'tensorflow'],
)
