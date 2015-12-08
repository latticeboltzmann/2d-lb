# 2d-lb

![Vortex Sheets](https://github.com/latticeboltzmann/2d-lb/blob/master/pictures/vortex_sheets.png)

An easy-to-read implementation of the D2Q9 Lattice-Boltzmann simulation in Python, Cython, and OpenCL created as a final
project for our CS205 class at Harvard. Our pyOpenCL code isn't too slow; we achieve roughly 325 MLUPS! 

For more details on the project and the Lattice Boltzmann technique, see our project website at

http://www.latticeboltzmann.us/

## Installation

To install, just use

    python setup.py install

or 

    pip install .

when you are in the directory with the setup.py file. Both should work. 

## How to use the code

To learn how to use the code, look in the "docs" folder and run the ipython notebooks there. The
[CS-205 movie IPython notebook](https://github.com/latticeboltzmann/2d-lb/blob/master/docs/cs205_movie.ipynb) is a
particularly fun place to start.

## Structure of the Code

### Packages

Within the LB_D2Q9 package, there are two subpackages: *OLD* and *dimensionless*. The LB_D2Q9 package also contains
our OpenCL kernels in the *D2Q9.cl* file.

#### OLD

The *OLD* subpackage is, unsurprisingly, old; we make no guarantees that the package will work. We have included it 
because it does not use dimensionless simulations, which can sometimes be useful.

#### Dimensionless

The dimensionless subpackage is well commented and should work on any computer as long as the requisite packages
are installed. This is the code we are submitting for our final project. It creates dimensionless Lattice Boltzmann
simulations. 

### Folders

The *docs* folder contains documents that should help the user see what our program can do and also teach the user
how to use our program. The *pictures* folder contains illustrative pictures of simulation results. The *testing* folder
contains a variety of old IPython notebooks that the authors used to test the software; we make no guarantee that these
will work as they are generally outdated.