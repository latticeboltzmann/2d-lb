# Intro

* Instead of calculating derivatives in a nonlocal manner, lattice boltzmann uses a variety of hoppers that dynamically
update the field locally
   * You track the position and velocity of each jumper and extract the fields from them.

* In between molecular dynamics simulations and continuum simulations, "mesoscale."

* This is perfect for an architecture like the GPU, where local operations are extremely efficient and massively 
parallelizable.

* Modern fluid dynamics simulation technique. Perfect for complicated geometries, avoids issues with singularities
* Works well up to a Reynold's number of about 10,000, so moderately turbulent flows...captures instabilities
* Easily apply different BC's

* Obeys all conservation laws such for the underlying flow, giving the simulations additional accuracy

# Performance Improvements: Second

* Created an equivalent simulation in python, cython, and opencl.
* cython was 10x faster than python, opencl was 650x faster than python on a GTX titan black video card. 
    * Huge performance increases, around 650x
*   

# Our simulations

* End with CS205 simulation lol
