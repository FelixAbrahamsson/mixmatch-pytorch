from setuptools import setup, find_packages


setup(
   name='mixmatch-pytorch',
   version='0.2.1',
   description='MixMatch for PyTorch',
   author='Felix Abrahamsson',
   author_email='FelixAbrahamsson@github.com',
   keywords='mixmatch holistic approach pytorch torch',
   packages=['mixmatch_pytorch'],
   install_requires=[
       'torch<=1.3',
   ],
)
