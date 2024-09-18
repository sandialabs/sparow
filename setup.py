"""
Script to generate the installer for forestlib
"""

import sys
import os
from setuptools import setup
import sys


def _find_packages(path):
    """
    Generate a list of nested packages
    """
    pkg_list = []
    if not os.path.exists(path):
        return []
    if not os.path.exists(path + os.sep + "__init__.py"):
        return []
    else:
        pkg_list.append(path)
    for root, dirs, files in os.walk(path, topdown=True):
        if root in pkg_list and "__init__.py" in files:
            for name in dirs:
                if os.path.exists(root + os.sep + name + os.sep + "__init__.py"):
                    pkg_list.append(root + os.sep + name)
    return [pkg for pkg in map(lambda x: x.replace(os.sep, "."), pkg_list)]


requires = ["Pyomo", "munch"]
packages = _find_packages("forestlib")

setup(
    name="forestlib",
    version="1.0",
    platforms=["any"],
    description="A forest of optimization capabilities",
    classifiers=[
        #'Development Status :: 5 - Production/Stable',
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        #'License :: OSI Approved :: BSD License',
        "Natural Language :: English",
        #'Operating System :: MacOS',
        #'Operating System :: Microsoft :: Windows',
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        #"Programming Language :: Python :: 3.7",
        #"Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=packages,
    install_requires=requires
)
