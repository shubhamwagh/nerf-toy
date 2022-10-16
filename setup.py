"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

setup(
    name='nerf_toy',
    version='0.0.2',
    description='A 2D toy illustration of Neural Radiance Fields',
    url='https://github.com/shubhamwagh/nerf-toy',
    long_description=README,
    long_description_content_type='text/markdown',
    license="MIT",
    platforms=['Ubuntu 20.04', 'Ubuntu 21.04', 'Windows'],
    author='Shubham Wagh',
    author_email='shubhamwagh48@gmail.com',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
    keywords='nerf, neural radiance fields, nerf2D, nerf toy, fourier, positional encoding',
    package_dir={"": "nerf_toy"},
    packages=find_packages(where="nerf_toy"),
    python_requires='>=3.6, <4',
    data_files=[],
    install_requires=['opencv-python', 'tensorflow', 'imageio', 'imageio-ffmpeg'],
)