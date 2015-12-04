# Intro

** Show fluid dynamics in action, i.e. von karmen sheets over an island**

* Fluid mechanics is all around us. Some problems in fluid mechanics, such as turbulence, are
  known as the last unsolved problem in classical physics.

* A relatively new method to simulate fluids was introduced in 1970 and significantly developed since then:
 the lattice boltzmann technique
 
* Some of its many advantages are to easily deal with complicated geometries, and the 
numerical scheme obeys all conservation laws, making the resulting simulations more reliable.
Works up until very turbulent flows.

* In between molecular dynamics simulations and continuum simulations, "mesoscale," and can consequently
capture more complicated physics that standard macroscopic simulation techniques such as finite volume and
finite element cannot.

* The algorithm involves tracking a variety of hoppers on a lattice that jump in a specified
direction. The fields of fluid mechanics, like pressure and velocity, are extracted from the concentration of 
the hoppers.

* The main advantage of the algorithm is that all derivatives which are typically handled in a nonlocal
manner become local; you only have to look up the value of the hoppers at that site. The simulation
is also more stable as a result.

* The lattice boltzmann technique is thus perfect to put on a GPU, as since everything is local,
the problem is massively parallelizable and efficient. 

## Performance Improvments: 