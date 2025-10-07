"""
Setup script for Mobile Panel Detection API
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mobile-panel-detector",
    version="2.4.0",
    author="Mobile Panel Detection Team",
    author_email="team@example.com",
    description="Mobile Panel Detection API with YOLO and OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mobile-panel-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mobile-panel-detector=mobile_panel_detector.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mobile_panel_detector": ["*.yaml", "*.yml", "*.json"],
    },
)
