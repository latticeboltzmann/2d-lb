# 2d-lb

![Vortex Sheets](https://github.com/latticeboltzmann/2d-lb/blob/master/pictures/vortex_sheets.png)

An easy-to-read implementation of the D2Q9 Lattice-Boltzmann simulation in Python, Cython, and OpenCL. Our pyOpenCL code isn't too slow; we achieve roughly 325 MLUPS!

To install, just use

    python setup.py install

or 

    pip install .

when you are in the directory with the setup.py file. Both should work. To learn how to use the code, look in the "docs" folder and run the ipython notebooks there. For more details on the project and the Lattice Boltzmann technique, see our project website at

http://www.latticeboltzmann.us/
