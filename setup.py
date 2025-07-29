#!/usr/bin/env python3
"""
Setup script for openIMIS Claim AI module
"""

from setuptools import setup, find_packages
import os


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="openimis-claim-ai",
    version="1.0.0",
    description="AI-powered claim processing module for OpenIMIS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="OpenIMIS Community",
    author_email="info@openimis.org",
    url="https://github.com/openimis/openimis-be-claim_ai_py",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Django",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="openimis, claim, ai, machine learning, healthcare, insurance",
    project_urls={
        "Bug Reports": "https://github.com/openimis/openimis-be-claim_ai_py/issues",
        "Source": "https://github.com/openimis/openimis-be-claim_ai_py",
        "Documentation": "https://openimis.readthedocs.io/",
    },
)
