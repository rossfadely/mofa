#!/usr/bin/env python

import os
import sys


try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    print "failed import"
    from distutils.core import setup
    from distutils.extension import setup, Extension
    setup, Extension

import numpy.distutils.misc_util

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

algorithms_ext = Extension("mofa._algorithms", ["mofa/_algorithms.c"])


setup(
    name="mofa",
    version="0.0.2a",
    author="Ross Fadely, David W. Hogg & Dan Foreman-Mackey",
    packages=["mofa"],
    url="https://github.com/rossfadely/mofa",
    license="MIT",  # Short name of license.
    description="Mixture of factor analyzers",
    long_description=open("README.rst").read(),
    package_data={"": ["LICENSE.rst", "AUTHORS.rst"]},
    ext_modules = [algorithms_ext],
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
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
