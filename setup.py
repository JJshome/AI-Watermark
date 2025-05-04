"""
AI Watermark - Setup Script

This module allows the installation of the AI Watermark tools as a package.
"""

from setuptools import setup, find_packages

setup(
    name="ai-watermark",
    version="0.1.0",
    description="Tools for embedding and detecting watermarks in AI-generated content",
    author="JJshome",
    author_email="107289883+JJshome@users.noreply.github.com",
    url="https://github.com/JJshome/AI-Watermark",
    packages=find_packages(),
    py_modules=[
        "audio_watermark",
        "video_watermark",
        "watermark_extractor",
        "audio_extract",
        "video_extract",
        "ai_watermark"
    ],
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-watermark=ai_watermark:main",
            "audio-watermark=audio_extract:main",
            "video-watermark=video_extract:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    long_description="""
    # AI Watermark

    Implementation of AI watermarking techniques for audio and video based on patent by Ucaretron Inc.
    This repository provides tools for embedding imperceptible watermarks in AI-generated content 
    to indicate confidence levels or identify AI-generated portions.

    ## Features

    * Audio watermarking using high-frequency embedding
    * Video watermarking with multiple methods (histogram, frame, pixel, interframe)
    * Extraction and analysis tools
    * Command-line interface
    * Programmatic API

    For more information, visit: https://github.com/JJshome/AI-Watermark
    """,
    long_description_content_type="text/markdown",
)
