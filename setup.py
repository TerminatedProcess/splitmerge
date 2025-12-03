#!/usr/bin/env python3
"""Setup script for splitmerge utility."""

from setuptools import setup

setup(
    name='splitmerge',
    version='1.0.0',
    description='Merge split safetensors files into a single file',
    author='Dev',
    py_modules=['splitmerge'],
    install_requires=[
        'safetensors>=0.4.0',
        'torch>=2.0.0',
        'packaging>=21.0',
        'numpy>=1.20.0',
    ],
    entry_points={
        'console_scripts': [
            'splitmerge=splitmerge:main',
        ],
    },
    python_requires='>=3.8',
)
