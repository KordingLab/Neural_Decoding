#! /usr/bin/env python
import setuptools  # noqa; we are using a setuptools namespace
# from numpy.distutils.core import setup
from setuptools import setup, Extension

descr = """A python package that includes many methods for decoding neural activity."""

DISTNAME = 'Neural Decoding'
DESCRIPTION = descr
MAINTAINER = 'Josh Glaser'
MAINTAINER_EMAIL = 'joshglaser88@gmail.com '
LICENSE = 'BSD 3-Clause License'
DOWNLOAD_URL = 'https://github.com/KordingLab/Neural_Decoding.git'
VERSION = '0.1.2.dev'
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          long_description_content_type= 'text/markdown',
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=['Neural_Decoding'],
          install_requires=requirements,
          )