from setuptools import setup, find_packages


setup(
   name='mixmatch_pytorch',
   version='0.1',
   description='Mixmatch for PyTorch',
   author='Felix Abrahamsson',
   author_email='FelixAbrahamsson@github.com',
   keywords='mixmatch holistic approach pytorch torch',
   packages=['mixmatch'],
   install_requires=[
       'torch<=1.3',
   ],
)
