#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='5gmdata',
      version='0.0.1',
      description='Datasets and code for machine learning in 5G mmWave MIMO systems involving mobility (5GMdata)', 
      author='LASSE',
      author_email='pedosb@gmail.com',
      url='https://github.com/lasseufpa/5gm-data',
      install_requires=['rwisimulation'],
      dependency_links=[
          'git+https://github.com/lasseufpa/5gm-rwi-simulation.git@master#egg=rwisimulation',
      ])
