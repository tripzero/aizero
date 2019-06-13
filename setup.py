#!/usr/bin/env python3

from setuptools import setup, find_packages

classifiers = """\
Development Status :: 4 - Beta
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)
Programming Language :: Python :: 3 :: Only
Topic :: System :: Hardware
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
"""

setup(name='aizero',
      version='1.3.0',
      description='Python module for simple machine learning',
      author='Kevron Rees',
      author_email='tripzero.kev@gmail.com',
      url='https://github.com/tripzero/ai',
      packages=["aizero"],
      license="LGPL Version 2.0",
      classifiers=filter(None, classifiers.split("\n"))
      )
