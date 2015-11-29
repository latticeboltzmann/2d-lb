from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
# import cython_gsl
import numpy as np

extensions = [
    Extension("LB_D2Q9.pipe_cython",
              sources=["LB_D2Q9/pipe_cython.pyx"])
]

setup(
    name='2d-lb',
    version='0.01',
    packages=['LB_D2Q9'],
    url='',
    license='',
    author='Bryan Weinstein, Matheus Fernandes',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules = cythonize(extensions, annotate=True, reload_support=True)
)
