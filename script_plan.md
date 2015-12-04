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

* Measure performance of simulations in MLUPS: million lattice updates per second
    * With openCL, we got about 325 MLUPS, python was about 0.5 MLUPS.
    * Commercial, fancy code gets about 1500 MLUPS, so we didn't do bad.

* Made our simulations dimensionless as suggested by the laws of fluid dynamics
* Verified that simulations at different resolutions converged to the correct scaled theoretical solution.


# Our simulations

* Confirmed different flow regimes around cylinder, i.e. vortex eddies
* Could simulate in wacky geometries easily, i.e. CS205 simulation

* In comparison to most code out there, our code is easy to understand and class based. Both of us have plans 
to build on this simulation package in the future.
