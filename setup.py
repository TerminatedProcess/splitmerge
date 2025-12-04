#!/usr/bin/env python3
"""Setup script for splitmerge utility."""

from setuptools import setup
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='splitmerge',
    version='1.0.0',
    description='Merge split safetensors model files into a single file',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Dev',
    author_email='',
    url='https://github.com/yourusername/splitmerge',
    license='MIT',
    keywords=['safetensors', 'ai', 'models', 'huggingface', 'merge', 'shards'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
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
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/splitmerge/issues',
        'Source': 'https://github.com/yourusername/splitmerge',
    },
)
