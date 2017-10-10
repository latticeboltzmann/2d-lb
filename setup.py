from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("LB_D2Q9.dimensionless.cython_dim",
              sources=["LB_D2Q9/dimensionless/cython_dim.pyx"],
              include_dirs = [np.get_include()]),

    Extension("LB_D2Q9.OLD.cython",
              sources=["LB_D2Q9/OLD/cython.pyx"],
              include_dirs = [np.get_include()])
]

setup(
    name='2d-lb',
    version='0.1',
    packages=['LB_D2Q9'],
    include_package_data=True,
    url='',
    license='',
    author='Bryan Weinstein, Matheus C. Fernandes',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules = cythonize(extensions, annotate=True),
    include_dirs = [np.get_include()],
    requires=['pyopencl', 'numpy', 'skimage', 'cython', 'seaborn']
)
