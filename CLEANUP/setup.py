from setuptools import setup, find_packages

setup(
    name="hexatic_order",
    version="0.1.0",
    packages=find_packages(),        # will pick up the hexatic_order package
    install_requires=[
        "numpy",
        "scipy"
    ],
)