# 2d-lb

An easy-to-read implementation of the D2Q9 Lattice-Boltzmann simulation. We have an implementation of the code in
Cython, Python, and OpenCL. Our OpenCL code isn't too slow; we achieve roughly 325 MLUPS!

To install, just use

    python setup.py install

or 

    pip install .

when you are in the directory. Both should work. To learn how to use the code, look in the "docs" folder and run 
the ipython notebooks there. 
