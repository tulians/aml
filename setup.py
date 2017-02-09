import os
try:
    from setuptools import setup, find_packages
except ImportError:
        from distutils.core import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="aml",
    version="0.2",
    author="Julian Ailan",
    author_email="jtulians@gmail.com",
    description=("Machine learning library aimed at code simplicity and"
                 " great execution speed."),
    license="MIT",
    keywords=["machinelearning", "neuralnetworks", "perceptron", "numpy"],
    url="https://github.com/tulians/aml",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
    long_description=read("README.md"),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
