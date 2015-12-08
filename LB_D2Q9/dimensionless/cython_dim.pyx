#cython: profile=True
#cython: linetrace=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import skimage as ski

##########################
##### D2Q9 parameters ####
##########################
w=np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,
            1./36.,1./36.,1./36.])      # weights for directions
cx=np.array([0,1,0,-1,0,1,-1,-1,1])     # direction vector for the x direction
cy=np.array([0,0,1,0,-1,1,1,-1,-1])     # direction vector for the y direction
cs=1/np.sqrt(3)                         # Speed of sound on the lattice
cs2 = cs**2                             # Speed of sound squared; a constant
cs22 = 2*cs2                            # Two times the speed of sound squared; another constant
cssq = 2.0/9.0                          # Another constant used in the update_feq method

w0 = 4./9.                              # Weight of stationary jumpers
w1 = 1./9.                              # Weight of horizontal and vertical jumpers
w2 = 1./36.                             # Weight of diagonal jumpers

NUM_JUMPERS = 9                         # Number of jumpers for the D2Q9 lattice: 9

class Pipe_Flow(object):
    """
    Simulates pipe flow using the D2Q9 lattice. Generally used to verify that our simulations were working correctly.
    For usage, see the docs folder.
    """

    def __init__(self, diameter=None, rho=None, viscosity=None, pressure_grad=1., pipe_length=None,
                 N=100, time_prefactor = 1.):
        """
        If an input parameter is physical, use "physical" units, i.e. a diameter could be specified in meters.

        :param diameter: Physical diameter of the 2-dimensional pipe.
        :param rho: Physical density of the fluid.
        :param viscosity: Physical kinematic density of the fluid.
        :param pressure_grad: Physical pressure gradient
        :param pipe_length: Physical length of the pipe
        :param N: Resolution of the simulation. As N increases, the simulation should become more accurate. N determines
                  how many grid points the characteristic length scale is discretized into
        :param time_prefactor: In order for a simulation to be accurate, in general, the dimensionless
                               space discretization delta_t ~ delta_x^2 (see http://wiki.palabos.org/_media/howtos:lbunits.pdf).
                               In our simulation, delta_t = time_prefactor * delta_x^2. delta_x is determined automatically
                               by N.
        """

        # Physical units
        self.phys_diameter = diameter
        self.phys_rho = rho
        self.phys_visc = viscosity
        self.phys_pressure_grad = pressure_grad
        self.phys_pipe_length = pipe_length

        # Get the characteristic length and time scales for the flow
        self.L = None   # Characteristic length scale
        self.T = None   # Characteristic time scale
        self.set_characteristic_length_time()
        print 'Characteristic L:', self.L
        print 'Characteristic T:', self.T

        # Initialize the reynolds number
        self.Re = self.L**2/(self.phys_visc*self.T**2)
        print 'Reynolds number:', self.Re

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = 1./N # How many squares characteristic length is broken into
        self.delta_t = time_prefactor * self.delta_x**2 # How many time iterations until the characteristic time, should be ~ \delta x^2

        # Initialize grid dimensions
        self.lx = None # Number of grid points in the x direction, ignoring the boundary
        self.ly = None # Number of grid points in the y direction, ignoring the boundary
        self.nx = None # Number of grid points in the x direction with the boundray
        self.ny = None # Number of grid points in the y direction with the boundary
        self.initialize_grid_dims()

        self.lb_viscosity = (self.delta_t/self.delta_x**2) * (1./self.Re) # Viscosity of the lattice boltzmann simulation

        # Get omega from lb_viscosity
        self.omega = (self.lb_viscosity/cs2 + 0.5)**-1. # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.

        ## Initialize hydrodynamic variables
        self.inlet_rho = None   # The density at the inlet...pressure boundary condition
        self.outlet_rho = None  # The density at the outlet...pressure boundary condition
        self.rho = None # The simulation's density field
        self.u = None # The simulation's velocity in the x direction (horizontal)
        self.v = None # The simulation's velocity in the y direction (vertical)
        self.init_hydro() # Create the hydrodynamic fields

        # Intitialize the underlying probablistic fields
        self.f=np.zeros((NUM_JUMPERS, self.nx, self.ny), dtype=np.float32) # Initialize f
        self.feq = np.zeros((NUM_JUMPERS, self.nx, self.ny), dtype=np.float32) # Initialize feq

        self.update_feq() # Based on the hydrodynamic fields, create feq
        self.init_pop() # Based on feq, create the hopping non-equilibrium fields


    def set_characteristic_length_time(self):
        """
        Based on the input parameters, set the characteristic length and time scales. Required to make
        the simulation dimensionless. See http://www.latticeboltzmann.us/home/model-verification for more details.
        For pipe flow, L is the physical diameter of the pipe, and T is the time it takes the fluid moving at its
        theoretical maximum to to move a distance of L.
        """
        self.L = self.phys_diameter
        self.T = (8*self.phys_rho*self.phys_visc)/(np.abs(self.phys_pressure_grad)*self.L)

    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """

        self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
        self.ly = self.N

        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary


    def init_hydro(self):
        """
        Based on the initial conditions, initialize the hydrodynamic fields, like density and velocity
        """
        nx = self.nx
        ny = self.ny

        # Create the inlet & outlet densities
        nondim_deltaP = (self.T**2/(self.phys_rho*self.L))*self.phys_pressure_grad
        # Obtain the difference in density (pressure) at the inlet & outlet
        delta_rho = self.nx*(self.delta_t**2/self.delta_x)*(1./cs2)*nondim_deltaP

        self.outlet_rho = 1.
        self.inlet_rho = 1. + np.abs(delta_rho)

        print 'inlet rho:' , self.inlet_rho
        print 'outlet rho:', self.outlet_rho


        self.rho = np.ones((nx, ny), dtype=np.float32)
        self.rho[0, :] = self.inlet_rho
        self.rho[self.lx, :] = self.outlet_rho # Is there a shock in this case? We'll see...
        for i in range(self.rho.shape[0]):
            self.rho[i, :] = self.inlet_rho - i*(self.inlet_rho - self.outlet_rho)/float(self.rho.shape[0])

        self.u = .0*np.random.randn(nx, ny) # Fluctuations in the fluid
        self.v = .0*np.random.randn(nx, ny) # Fluctuations in the fluid


    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Note that this function was based on Sauro Succi's fortran code (he figured out an efficient way to do this).
        """

        u = self.u
        v = self.v
        rho = self.rho
        feq = self.feq

        ul = u/cs2
        vl = v/cs2
        uv = ul*vl
        usq = u*u
        vsq = v*v
        sumsq  = (usq+vsq)/cs22
        sumsq2 = sumsq*(1.-cs2)/cs2
        u2 = usq/cssq
        v2 = vsq/cssq

        feq[0, :, :] = w0*rho*(1. - sumsq)
        feq[1, :, :] = w1*rho*(1. - sumsq  + u2 + ul)
        feq[2, :, :] = w1*rho*(1. - sumsq  + v2 + vl)
        feq[3, :, :] = w1*rho*(1. - sumsq  + u2 - ul)
        feq[4, :, :] = w1*rho*(1. - sumsq  + v2 - vl)
        feq[5, :, :] = w2*rho*(1. + sumsq2 + ul + vl + uv)
        feq[6, :, :] = w2*rho*(1. + sumsq2 - ul + vl - uv)
        feq[7, :, :] = w2*rho*(1. + sumsq2 - ul - vl + uv)
        feq[8, :, :] = w2*rho*(1. + sumsq2 + ul - vl - uv)

    def init_pop(self):
        """Based on feq, create the initial population of jumpers."""

        feq = self.feq
        nx = self.nx
        ny = self.ny

        self.f = feq.copy()
        # We now slightly perturb f
        amplitude = .001
        perturb = (1. + amplitude*np.random.randn(nx, ny))
        self.f *= perturb

    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        We cythonized this function so that it was fast.
        """

        cdef int lx = self.lx
        cdef int ly = self.ly
        cdef int i, j

        farr = self.f

        # INLET: constant pressure!
        farr[1, 0, 1:ly] = farr[3, 0, 1:ly] + (2./3.)*self.inlet_rho*self.u[0, 1:ly]
        farr[5, 0, 1:ly] = -.5*farr[2,0,1:ly]+.5*farr[4, 0, 1:ly]+farr[7, 0, 1:ly] + (1./6.)*self.u[0, 1:ly]*self.inlet_rho
        farr[8, 0, 1:ly] = .5*farr[2,0,1:ly]-.5*farr[4, 0, 1:ly]+farr[6, 0, 1:ly] + (1./6.)*self.u[0, 1:ly]*self.inlet_rho

        # OUTLET: constant pressure!
        farr[3, lx, 1:ly] = farr[1, lx, 1:ly] - (2./3.)*self.outlet_rho*self.u[lx,1:ly]
        farr[6, lx, 1:ly] = -.5*farr[2,lx,1:ly]+.5*farr[4,lx,1:ly]+farr[8,lx,1:ly]-(1./6.)*self.u[lx,1:ly]*self.outlet_rho
        farr[7, lx, 1:ly] = .5*farr[2,lx,1:ly]-.5*farr[4,lx,1:ly]+farr[5,lx,1:ly]-(1./6.)*self.u[lx,1:ly]*self.outlet_rho

        cdef float[:, :, :] f = self.f
        cdef float inlet_rho = self.inlet_rho
        cdef float outlet_rho = self.outlet_rho

        with nogil:
            # NORTH solid
            for i in range(1, lx): # Bounce back
                f[4,i,ly] = f[2,i,ly]
                f[8,i,ly] = f[6,i,ly]
                f[7,i,ly] = f[5,i,ly]
            # SOUTH solid
            for i in range(1, lx):
                f[2,i,0] = f[4,i,0]
                f[6,i,0] = f[8,i,0]
                f[5,i,0] = f[7,i,0]

            ### Corner nodes: Tricky & a huge pain ###
            # BOTTOM INLET
            f[1, 0, 0] = f[3, 0, 0]
            f[2, 0, 0] = f[4, 0, 0]
            f[5, 0, 0] = f[7, 0, 0]
            f[6, 0, 0] = .5*(-f[0,0,0]-2*f[3,0,0]-2*f[4,0,0]-2*f[7,0,0]+inlet_rho)
            f[8, 0, 0] = .5*(-f[0,0,0]-2*f[3,0,0]-2*f[4,0,0]-2*f[7,0,0]+inlet_rho)

            # TOP INLET
            f[1, 0, ly] = f[3, 0, ly]
            f[4, 0, ly] = f[2, 0, ly]
            f[5, 0, ly] = .5*(-f[0,0,ly]-2*f[2,0,ly]-2*f[3,0,ly]-2*f[6,0,ly]+inlet_rho)
            f[7, 0, ly] = .5*(-f[0,0,ly]-2*f[2,0,ly]-2*f[3,0,ly]-2*f[6,0,ly]+inlet_rho)
            f[8, 0, ly] = f[6, 0, ly]

            # BOTTOM OUTLET
            f[3, lx, 0] = f[1, lx, 0]
            f[2, lx, 0] = f[4, lx, 0]
            f[6, lx, 0] = f[8, lx, 0]
            f[5, lx, 0] = .5*(-f[0,lx,0]-2*f[1,lx,0]-2*f[4,lx,0]-2*f[8,lx,0]+outlet_rho)
            f[7, lx, 0] = .5*(-f[0,lx,0]-2*f[1,lx,0]-2*f[4,lx,0]-2*f[8,lx,0]+outlet_rho)

            # TOP OUTLET
            f[3, lx, ly] = f[1, lx, ly]
            f[4, lx, ly] = f[2, lx, ly]
            f[6, lx, ly] = .5*(-f[0,lx,ly]-2*f[1,lx,ly]-2*f[2,lx,ly]-2*f[5,lx,ly]+outlet_rho)
            f[7, lx, ly] = f[5, lx, ly]
            f[8, lx, ly] = .5*(-f[0,lx,ly]-2*f[1,lx,ly]-2*f[2,lx,ly]-2*f[5,lx,ly]+outlet_rho)

    def move(self):
        """
        Move all other jumpers than those on the boundary. We cythonized this function; it was a bottleneck in
        pure python code.
        """
        cdef float[:, :, :] f = self.f
        cdef int lx = self.lx
        cdef int ly = self.ly

        cdef int i, j

        # This can't be parallelized without making a copy...order of loops is super important!
        with nogil:
            for j in range(ly,0,-1): # Up, up-left
                for i in range(0, lx):
                    f[2,i,j] = f[2,i,j-1]
                    f[6,i,j] = f[6,i+1,j-1]
            for j in range(ly,0,-1): # Right, up-right
                for i in range(lx,0,-1):
                    f[1,i,j] = f[1,i-1,j]
                    f[5,i,j] = f[5,i-1,j-1]
            for j in range(0,ly): # Down, right-down
                for i in range(lx,0,-1):
                    f[4,i,j] = f[4,i,j+1]
                    f[8,i,j] = f[8,i-1,j+1]
            for j in range(0,ly): # Left, left-down
                for i in range(0, lx):
                    f[3,i,j] = f[3,i+1,j]
                    f[7,i,j] = f[7,i+1,j+1]


    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables
        """
        f = self.f

        rho = self.rho
        rho[:, :] = np.sum(f, axis=0)
        inverse_rho = 1./self.rho

        u = self.u
        v = self.v

        u[:, :] = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])*inverse_rho
        v[:, :] = (f[5]+f[2]+f[6]-f[7]-f[4]-f[8])*inverse_rho

        # 0 velocity on walls
        u[:, 0] = 0
        u[:, self.ly] = 0
        v[:, 0] = 0
        v[:, self.ly] = 0

        # Deal with boundary conditions...have to specify pressure
        lx = self.lx

        rho[0, :] = self.inlet_rho
        rho[lx, :] = self.outlet_rho
        # INLET
        u[0, :] = 1 - (f[0, 0, :]+f[2, 0, :]+f[4, 0, :]+2*(f[3, 0, :]+f[6, 0, :]+f[7, 0, :]))/self.inlet_rho

        # OUTLET
        u[lx, :] = -1 + (f[0, lx, :]+f[2, lx, :]+f[4, lx, :]+2*(f[1, lx, :]+f[5, lx, :]+f[8, lx, :]))/self.outlet_rho


    def collide_particles(self):
        """
        Relax the nonequilibrium f fields towards their equilibrium feq. Depends on omega.
        """
        f = self.f
        feq = self.feq
        omega = self.omega

        self.f[:, :, :] = f*(1.-omega)+omega*feq

    def run(self, num_iterations):
        """
        Run the simulation for num_iterations. Be aware that the same number of iterations does not correspond
        to the same non-dimensional time passing, as delta_t, the time discretization, will change depending on
        your resolution.

        :param num_iterations: The number of iterations to run
        """
        for cur_iteration in range(num_iterations):
            self.move_bcs() # We have to udpate the boundary conditions first, or we are in trouble.
            self.move() # Move all jumpers
            self.update_hydro() # Update the hydrodynamic variables
            self.update_feq() # Update the equilibrium fields
            self.collide_particles() # Relax the nonequilibrium fields

    def get_fields(self):
        """
        :return: Returns a dictionary of all fields. More useful for the OpenCL code, where we have to transfer
                data from the GPU to the CPU.
        """

        results={}
        results['f'] = self.f
        results['u'] = self.u
        results['v'] = self.v
        results['rho'] = self.rho
        results['feq'] = self.feq
        return results

    def get_nondim_fields(self):
        """
        :return: Returns a dictionary of the fields scaled so that they are in non-dimensional form.
        """
        fields = self.get_fields()

        fields['u'] *= self.delta_x/self.delta_t
        fields['v'] *= self.delta_x/self.delta_t

        return fields

    def get_physical_fields(self):
        """
        :return: Returns a dictionary of the fields scaled so that they are in physical form; this is probably what
                 most users are interested in.
        """
        fields = self.get_nondim_fields()

        fields['u'] *= (self.L/self.T)
        fields['v'] *= (self.L/self.T)

        return fields

class Pipe_Flow_Cylinder(Pipe_Flow):
    """
    A subclass of the Pipe Flow class that simulates fluid flow around a cylinder. This class can also be "hacked"
    in its current state to simulate flow around arbitrary obstacles. See
    https://github.com/latticeboltzmann/2d-lb/blob/master/docs/cs205_movie.ipynb for an example of how to do so.
    """

    def set_characteristic_length_time(self):
        """
        Sets the characteristic length and time scale. For the cylinder, the characteristic length scale is
        the cylinder radius. The characteristic time scale is the time it takes the fluid in the pipe moving at its
        theoretical maximum to move over the cylinder.
        """
        self.L = self.phys_cylinder_radius
        self.T = (8*self.phys_rho*self.phys_visc*self.L)/(np.abs(self.phys_pressure_grad)*self.phys_diameter**2)

    def initialize_grid_dims(self):
        """Initializes the grid, like above, but also initializes an appropriate mask of the obstacle."""

        self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
        self.ly = int(np.ceil((self.phys_diameter / self.L)*self.N))

        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

        ## Initialize the obstacle mask
        self.obstacle_mask = np.zeros((self.nx, self.ny), dtype=np.bool, order='F')

        # Initialize the obstacle in the correct place
        x_cylinder = self.N * self.phys_cylinder_center[0]/self.L
        y_cylinder = self.N * self.phys_cylinder_center[1]/self.L

        circle = ski.draw.circle(x_cylinder, y_cylinder, self.N)
        self.obstacle_mask[circle[0], circle[1]] = True

    def __init__(self, cylinder_center = None, cylinder_radius=None, **kwargs):
        """
        :param cylinder_center: The center of the cylinder in physical units.
        :param cylinder_radius: The raidus of the cylinder in physical units.
        :param kwargs: All keyword arguments required to initialize the pipe-flow class.
        """

        assert (cylinder_center is not None) # If the cylinder does not have a center, this code will explode
        assert (cylinder_radius is not None) # If there are no obstacles, this will definitely not run.

        self.phys_cylinder_center = cylinder_center # Center of the cylinder in physical units
        self.phys_cylinder_radius = cylinder_radius # Radius of the cylinder in physical units

        self.obstacle_mask = None # A boolean mask of the location of the obstacle
        super(Pipe_Flow_Cylinder, self).__init__(**kwargs) # Initialize the superclass
        self.obstacle_pixels = np.where(self.obstacle_mask) # A list of x and y coordinates of the obstacle

    def init_hydro(self):
        """
        Overrides the init_hydro method in Pipe_Flow.
        """
        super(Pipe_Flow_Cylinder, self).init_hydro()
        # The velocity inside the obstacle must be zero.
        self.u[self.obstacle_mask] = 0
        self.v[self.obstacle_mask] = 0

    def update_hydro(self):
        """
        Overrides the init_hydro method in Pipe_Flow.
        """
        super(Pipe_Flow_Cylinder, self).update_hydro()
        # The velocity inside the obstacle must be zero
        self.u[self.obstacle_mask] = 0
        self.v[self.obstacle_mask] = 0

    def move_bcs(self):
        """
        Overrides the move_bcs method in Pipe_Flow
        """
        super(Pipe_Flow_Cylinder, self).move_bcs()

        # Now bounceback on the obstacle
        cdef long[:] x_list = self.obstacle_pixels[0]
        cdef long[:] y_list = self.obstacle_pixels[1]
        cdef int num_pixels = y_list.shape[0]

        cdef float[:, :, :] f = self.f

        cdef float old_f0, old_f1, old_f2, old_f3, old_f4, old_f5, old_f6, old_f7, old_f8
        cdef int i
        cdef long x, y

        with nogil:
            for i in range(num_pixels):
                x = x_list[i]
                y = y_list[i]

                old_f0 = f[0, x, y]
                old_f1 = f[1, x, y]
                old_f2 = f[2, x, y]
                old_f3 = f[3, x, y]
                old_f4 = f[4, x, y]
                old_f5 = f[5, x, y]
                old_f6 = f[6, x, y]
                old_f7 = f[7, x, y]
                old_f8 = f[8, x, y]

                # Bounce back everywhere!
                # left right
                f[1, x, y] = old_f3
                f[3, x, y] = old_f1
                # up down
                f[2, x, y] = old_f4
                f[4, x, y] = old_f2
                # up-right
                f[5, x, y] = old_f7
                f[7, x, y] = old_f5

                # up-left
                f[6, x, y] = old_f8
                f[8, x, y] = old_f6

### Matt Stuff ###

#TID

# class Velocity_Inlet_Cylinder(Pipe_Flow_Cylinder):
#
#     def __init__(self, *args, inlet_velocity=None, **kwargs):
#
#         assert inlet_velocity is not None
#         self.phys_inlet_velocity = inlet_velocity
#         self.dim_inlet_velocity = None
#         self.lb_inlet_velocity = None
#
#         super(Velocity_Inlet_Cylinder, self).__init__(**kwargs)
#
#
#     def set_characteristic_length_time(self):
#         """Necessary for subclassing"""
#         self.L = self.phys_cylinder_radius
#         self.T = self.phys_cylinder_radius/self.phys_inlet_velocity
#
#         self.dim_inlet_velocity = (self.T/self.L) * self.phys_inlet_velocity
#
#     def initialize_grid_dims(self):
#         """Necessary for subclassing"""
#
#         self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
#         self.ly = int(np.ceil((self.phys_diameter / self.L)*self.N))
#
#         self.nx = self.lx + 1 # Total size of grid in x including boundary
#         self.ny = self.ly + 1 # Total size of grid in y including boundary
#
#         ## Initialize the obstacle mask
#         self.obstacle_mask = np.zeros((self.nx, self.ny), dtype=np.bool, order='F')
#
#         # Initialize the obstacle in the correct place
#         x_cylinder = self.N * self.phys_cylinder_center[0]/self.L
#         y_cylinder = self.N * self.phys_cylinder_center[1]/self.L
#
#         circle = ski.draw.circle(x_cylinder, y_cylinder, self.N)
#         self.obstacle_mask[circle[0], circle[1]] = True
#
#     def move_bcs(self):
#         """This is slow; cythonizing makes it fast."""
#
#         cdef int lx = self.lx
#         cdef int ly = self.ly
#
#         cdef float u_w = self.lb_inlet_velocity
#         cdef float u_e = self.lb_inlet_velocity
#
#         cdef float[:,:,:] farr = self.f
#         cdef float[:] rho_w = self.rho[0, 1:ly]
#         cdef float[:] rho_e = self.rho[lx, 1:ly]
#
#         cdef float[:,:,:] f = self.f
#         cdef int ii
#
#         for ii in range(1,ly):
#             # INLET: imposed velocity of u_w in the x direction and 0 in the y direction
#
#             farr[1, 0, ii] = farr[3,0,ii] + (2./3.)*rho_w[ii]*u_w
#             farr[5, 0, ii] = farr[7,0,ii] - (1./2.)*(farr[2,0,ii]-farr[4,0,ii]) + (1./6.)*rho_w[ii]*u_w
#             farr[8, 0, ii] = farr[6,0,ii] + (1./2.)*(farr[2,0,ii]-farr[4,0,ii]) + (1./6.)*rho_w[ii]*u_w
#
#             # OUTLET: imposed velocity of u_w in the x direction and 0 in the y direction
#             rho_e[ii] = (1./(1.+u_e))*(farr[0,lx,ii]+farr[2,lx,ii]+farr[4,lx,ii]+2.*(farr[1,lx,ii]+farr[5,lx,ii]+farr[8,lx,ii]))
#
#             farr[3, lx, ii] = farr[1,lx,ii] - (2./3.)*rho_e[ii]*u_e
#             farr[7, lx, ii] = farr[5,lx,ii] + (1./2.)*(farr[2,lx,ii]-farr[4,lx,ii]) - (1./6.)*rho_e[ii]*u_e
#             farr[6, lx, ii] = farr[8,lx,ii] - (1./2.)*(farr[2,lx,ii]-farr[4,lx,ii]) - (1./6.)*rho_e[ii]*u_e
#
#         for ii in range(0,lx+1):
#             # NORTH periodic
#             # update the values of f at the top with those from the bottom
#             f[4,ii,ly] = f[4,ii,0]
#             f[8,ii,ly] = f[8,ii,0]
#             f[7,ii,ly] = f[7,ii,0]
#             # SOUTH periodic
#             #update the values of f at the bottom with those from the top
#             f[2,ii,0] = f[2,ii,ly]
#             f[6,ii,0] = f[6,ii,ly]
#             f[5,ii,0] = f[5,ii,ly]
#
#         # Now bounceback on the obstacle
#         cdef long[:] x_list = self.obstacle_pixels[0]
#         cdef long[:] y_list = self.obstacle_pixels[1]
#         cdef int num_pixels = y_list.shape[0]
#
#         cdef float old_f0, old_f1, old_f2, old_f3, old_f4, old_f5, old_f6, old_f7, old_f8
#         cdef int j
#         cdef long x, y
#
#         with nogil:
#             for j in range(num_pixels):
#                 x = x_list[j]
#                 y = y_list[j]
#
#                 old_f0 = f[0, x, y]
#                 old_f1 = f[1, x, y]
#                 old_f2 = f[2, x, y]
#                 old_f3 = f[3, x, y]
#                 old_f4 = f[4, x, y]
#                 old_f5 = f[5, x, y]
#                 old_f6 = f[6, x, y]
#                 old_f7 = f[7, x, y]
#                 old_f8 = f[8, x, y]
#
#                 # Bounce back everywhere!
#                 # left right
#                 f[1, x, y] = old_f3
#                 f[3, x, y] = old_f1
#                 # up down
#                 f[2, x, y] = old_f4
#                 f[4, x, y] = old_f2
#                 # up-right
#                 f[5, x, y] = old_f7
#                 f[7, x, y] = old_f5
#
#                 # up-left
#                 f[6, x, y] = old_f8
#                 f[8, x, y] = old_f6
#
#     def init_hydro(self):
#         """We have to initialize everything is dimensionless units!"""
#
#         nx = self.nx
#         ny = self.ny
#
#         self.rho = np.ones((nx, ny), dtype=np.float32)
#
#         self.u = np.zeros((nx, ny)) #initializing the fluid velocity matrix
#         self.v = np.zeros((nx, ny))
#
#         self.lb_inlet_velocity = self.dim_inlet_velocity * (self.delta_t/self.delta_x)
#
#         self.u[:,:]=self.lb_inlet_velocity
#         self.v[:,:]=0
#
#         # Zero velocity in the obstacles...not taken care of as we don't call super
#         self.u[self.obstacle_mask] = 0
#         self.v[self.obstacle_mask] = 0
#
#     def update_hydro(self):
#         f = self.f
#
#         rho = self.rho
#         rho[:, :] = np.sum(f, axis=0)
#         inverse_rho = 1./self.rho
#
#         u = self.u
#         v = self.v
#
#         u[:, :] = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])*inverse_rho
#         v[:, :] = (f[5]+f[2]+f[6]-f[7]-f[4]-f[8])*inverse_rho
#
#         # Deal with boundary conditions...
#         lx = self.lx
#         ly = self.ly
#
#         u_w=self.lb_inlet_velocity
#         u_e=self.lb_inlet_velocity
#
#         # INLET: define the density and prescribe the velocity
#         u[0, 1:ly] = u_w
#         rho[0, 1:ly] = (1./(1.-u_w))*(f[0,0,1:ly]+f[2,0,1:ly]+f[4,0,1:ly]+2.*(f[3,0,1:ly]+f[6,0,1:ly]+f[7,0,1:ly]))
#         # OUTLET: define the density and prescribe the velocity
#         u[lx, 1:ly] =u_e
#         rho[lx, 1:ly] = (1./(1.+u_e))*(f[0,lx,1:ly]+f[2,lx,1:ly]+f[4,lx,1:ly]+2.*(f[1,lx,1:ly]+f[5,lx,1:ly]+f[8,lx,1:ly]))
#
#         # Zero velocity in the obstacle
#         self.u[self.obstacle_mask] = 0
#         self.v[self.obstacle_mask] = 0
