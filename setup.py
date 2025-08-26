from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satchange",
    version="0.1.0",
    author="SatChange Team",
    author_email="team@satchange.dev",
    description="A CLI tool for detecting temporal changes in satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Updated package metadata and dependencies
    url="https://github.com/satchange/satchange",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "satchange=satchange.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "satchange": ["templates/*.html"],
    },
)