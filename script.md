# Script

MF: Hello everyome, my name is Matt Fernandes

BW: and I am Bryan Weinstein. 

MF: In this video we will be showing you a parallel implementation of the Lattice Boltzmann Method modeling a continuum scale fluid dynamics problem. First, Bryan, explain to us, why are we using the Lattice Boltzman method and what in the world is it?

BW: Matt -- the Lattice boltzmann is a method that uses local dynamic hoppers that interact with each other allowing us to calculate the field flow localy. We do this by calculating the density of these hoppers and update its location based on a stochastic approximation.  In turn, we are able to eliminate the nasty non-linearity implicit in the analytical continuum model known as the Navier-Stokes equation. 

MF: But how can we benefit from using multiple threads to solve this problem? What are advantages of using the lattice boltzmann method?

BW: The mothod is perfect for parallelization and can make great use of an architecture like the GPU because there is no limit in how much you can divide up the work between each processor. Except for potentially along the boundaries where we impose different boundary conditions, but even there most of times it is possible to parallelize. 

MF: Can we look at complex geometries as well? 

BW: Yes, as matter of fact we can use almost anything we want, as long as we refine the discretization well enough. The method is even great for computing flows around sharp corners, which in other methos would run into singularities. 

MF: Wow, how awesome is that? Now let's talk about performance. What were the different stages of our project and how did each stage perform as far as computational efficiency?

BW: Matt, as the graph shows, we were able to get performance improvements of x between Python and Cython, but what was really impressive was that we were able to get x times the MLUPS, which is ..., between Python and using OpenCL. These computations take in the order of ... secods vs the original ... minuites for the original naive code.

MF: But how accurate are these simulations?

BW: They are ssurprisingly very accurate. If we comapre against analytical solutios we see that the behaviour is the same. Also, if we compare the dyanimical solution to a COMOSOL simulation we get the same behavior. 

MF: Now let's look at the simulations results in action. Bryan, explain to what are the boundary conditions here, and what are we looking at.

BW: 

# Outline

## Intro

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

## Performance Improvements: Second

* Created an equivalent simulation in python, cython, and opencl.
* cython was 10x faster than python, opencl was 650x faster than python on a GTX titan black video card. 
    * Huge performance increases, around 650x

* Measure performance of simulations in MLUPS: million lattice updates per second
    * With openCL, we got about 325 MLUPS, python was about 0.5 MLUPS.
    * Commercial, fancy code gets about 1500 MLUPS, so we didn't do bad.

* Made our simulations dimensionless as suggested by the laws of fluid dynamics
* Verified that simulations at different resolutions converged to the correct scaled theoretical solution.


## Our simulations

* Confirmed different flow regimes around cylinder, i.e. vortex eddies
* Could simulate in wacky geometries easily, i.e. CS205 simulation

* In comparison to most code out there, our code is easy to understand and class based. Both of us have plans 
to build on this simulation package in the future.
