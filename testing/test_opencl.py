import seaborn as sns
from LB_D2Q9 import pipe_opencl as lb
import numpy as np

#### Input to the simulation in SI. ######
diameter = 4. # meters
length = 10. # meters

deltaP = -0.2

dr = 0.01
dt = 4.

viscosity = 10.**-6. # More viscous = easier to simulate!

##### Derived parameters ######
print 'Desired lb_viscosity:' , viscosity
#Re = (input_velocity * diameter)/lb_viscosity
#print 'Desired Re', Re

# Re is set by diameter, input_velocity, and lb_viscosity
# Adjust dr so that mach number doers not explode!
print
print '###### Simulation Parameters #########'
print

# Solve for the desired omega...make sure it's in the correct range.


ly = int(np.ceil(diameter/dr))
lx = int(np.ceil(length/dr))
print 'ly', ly
print 'lx', lx

nx = lx + 1
ny = ly + 1

omega = .5 + (3*dt*viscosity)/(dr**2)

print 'omega:' , omega

assert (omega > 0.5) and (omega < 1)

obstacle_size=.25 # meters

sim = lb.Pipe_Flow(lx=lx, ly=ly, dr=dr, dt=dt, omega=omega,
                  deltaP = deltaP)

# U should be on the order of 0.1, 0.2 in the simulation!
#print 'u_max in simulation:' , np.max(sim.u)