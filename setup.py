#!/usr/bin/env python
import os
from numpy.distutils.core import setup

setup(name='hlm_basic',
      version='0.1',
      author='faruk gurbuz',
      author_email='gurbuzf@hotmail.com',
      url='https://github.com/gurbuzf',
      packages=['hlm_basic', 'ga'],
      package_data={'data':[]},
      description='Simple HLM implementation with or without dams',
      long_description=open('README.md').read(),
      install_requires=[
       "scipy >= 1.4.1",
   ],
     )