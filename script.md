# Intro

** Show fluid dynamics in action, i.e. von karmen sheets over an island**

* Fluid mechanics is all around us. Some problems in fluid mechanics, such as turbulence, are
  known as the last unsolved problem in classical physics.

** Show picture of boltzmann and a lattice **

* A relatively new method to simulate fluids was introduced in 1970 and significantly developed since then:
 the lattice boltzmann technique
 
* Some of its many advantages are to easily deal with complicated geometries, and the 
numerical scheme obeys all conservation laws, making the resulting simulations more reliable.
Works up until very turbulent flows.

* In between molecular dynamics simulations and continuum simulations, "mesoscale," and can consequently
capture more complicated physics that standard macroscopic simulation techniques such as finite volume and
finite element cannot.

** Show lattice with hoppers on it **

* The algorithm involves tracking a variety of hoppers on a lattice that jump in a specified
direction. The fields of fluid mechanics, like pressure and velocity, are extracted from the concentration of 
the hoppers.

* The main advantage of the algorithm is that all derivatives which are typically handled in a nonlocal
manner become local; you only have to look up the value of the hoppers at that site. The simulation
is also more stable as a result.

* The lattice boltzmann technique is thus perfect to put on a GPU, as since everything is local,
the problem is massively parallelizable and efficient. 

# What we did

* We created a lattice boltzmann simulation of pipe flow around obstacles in python, cython, 
and opencl (the last two being what we learned in CS205).
* We both plan to use the lattice boltzmann technique in our research in the future, so we 
wanted to make our own fast repository that we truly understood.

** Show image of pipe flow verification **

* We verified our simulations by looking at the solution to the standard pipe flow problem.
* Made our simulations dimensionless as suggested by the laws of fluid dynamics
* Verified that simulations at different resolutions converged to the correct scaled theoretical solution.

** Show image of speedup **

* cython was 10x faster than python, opencl was 650x faster than python on a GTX titan black video card. 
    * Huge performance increases, around 650x
    
* Measure performance of simulations in MLUPS: million lattice updates per second
    * With openCL, we got about 325 MLUPS, python was about 0.5 MLUPS.
    * Commercial, fancy code gets about 1500 MLUPS, so we didn't do bad.
    * Our code could be further optimized as well in an obvious manner by about 10%. The other 
    optimization steps are less clear.
    
# Our simulations

* Confirmed different flow regimes around cylinder, i.e. vortex eddies. The buildup of flow, then 
the stretching, and then saw the vortices.
* Could simulate in wacky geometries easily, i.e. CS205 simulation. Made physical sense.

* In comparison to most code out there, our code is easy to understand and class based. Both of us have plans 
to build on this simulation package in the future, perhaps by understanding how populations grow
when advected by flow.
