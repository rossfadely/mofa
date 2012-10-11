#!/usr/bin/env python

import os
import sys
import mofa

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="mofa",
    version=mofa.__version__,
    author="Ross Fadely, David W. Hogg & Dan Foreman-Mackey",
    packages=["mofa"],
    url="https://github.com/rossfadely/mofa",
    license="MIT",  # Short name of license.
    description=mofa.__doc__,
    long_description=open("README.rst").read(),
    package_data={"": ["LICENSE.rst", "AUTHORS.rst"]},
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    classifiers=[
        # "Development Status :: 1 - Planning",
        # "Development Status :: 2 - Pre-Alpha",
        "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        # "Development Status :: 6 - Mature",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Choose a license from:
        #       http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # "License :: OSI Approved :: BSD License",
        # "License :: OSI Approved :: MIT License",
        # "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
