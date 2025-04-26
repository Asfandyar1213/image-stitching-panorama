from setuptools import setup, find_packages

setup(
    name="image-stitching-panorama",
    version="1.0.0",
    description="Advanced image stitching system for panorama generation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YourUsername/image-stitching-panorama",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.3",
        "gradio>=3.50.2",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    keywords="image-stitching, panorama, computer-vision, opencv, gradio",
) 