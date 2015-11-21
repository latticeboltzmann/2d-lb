#cython: profile=False
#cython: boundscheck=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

##########################
##### D2Q9 parameters ####
##########################
w=np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,
            1./36.,1./36.,1./36.]) # weights for directions
cx=np.array([0,1,0,-1,0,1,-1,-1,1]) # direction vector for the x direction
cy=np.array([0,0,1,0,-1,1,1,-1,-1]) # direction vector for the y direction
cs=1/np.sqrt(3)
cs2 = cs**2
cs22 = 2*cs2
cssq = 2.0/9.0

w0 = 4./9.
w1 = 1./9.
w2 = 1./36.

NUM_JUMPERS = 9

class Pipe_Flow(object):
    """2d pipe flow with D2Q9"""

    def __init__(self, omega=.99, lx=400, ly=400, dr=None, dt = None, deltaP=None):
        ### User input parameters
        self.lx = lx # Grid not including boundary in x
        self.ly = ly # Grid not including boundary in y

        if dr is None: self.dr = 1.
        else: self.dr = dr
        if dt is None: self.dt = 1.
        else: self.dt = dt
        if deltaP is None: self.deltaP = -1.0
        else: self.deltaP = deltaP

        self.omega = omega

        ## Everything else
        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

        # Based on deltaP, set rho at the edges, as P = rho/3
        self.inlet_rho = 1.0
        self.outlet_rho = cs2*self.deltaP + self.inlet_rho # deltaP is negative!


        ## Initialize hydrodynamic variables
        self.rho = None # Density
        self.u = None # Horizontal flow
        self.v = None # Vertical flow
        self.init_hydro()

        # Intitialize the underlying probablistic fields
        self.f=np.zeros((NUM_JUMPERS, self.nx, self.ny), dtype=np.float32) # initializing f
        self.feq = np.zeros((NUM_JUMPERS, self.nx, self.ny), dtype=np.float32)

        self.update_feq()
        self.init_pop()

        # Based on initial parameters, determine dimensionless numbers
        self.viscosity = None
        self.Re = None
        self.Ma = None
        self.set_dimensionless_nums()

        print 'Viscosity:' , self.viscosity
        print 'Re:' , self.Re
        print 'Ma:' , self.Ma

    def set_dimensionless_nums(self):
        self.viscosity = (self.dr**2/(3*self.dt))*(self.omega-0.5)

        # Get the reynolds number
        #U = self.input_velocity
        #L = self.ly*self.dr
        #self.Re = U*L/self.viscosity

        # To get the mach number...
        #self.Ma = (self.dr/(L*np.sqrt(3)))*(self.omega-.5)*self.Re


    def init_hydro(self):
        nx = self.nx
        ny = self.ny

        self.rho = np.ones((nx, ny), dtype=np.float32)
        self.rho[0, :] = self.inlet_rho
        self.rho[self.lx, :] = self.outlet_rho # Is there a shock in this case? We'll see...
        for i in range(self.rho.shape[0]):
            self.rho[i, :] = self.inlet_rho - i*(self.inlet_rho - self.outlet_rho)/float(self.rho.shape[0])

        self.u = .1*np.ones((nx, ny), dtype=np.float32) + .01*np.random.randn(nx, ny)
        self.v = .1*np.ones((nx, ny), dtype=np.float32) + .01*np.random.randn(nx, ny)


    def update_feq(self):
        """Taken from sauro succi's code. This will be super easy to put on the GPU."""

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

    def update_hydro(self):
        f = self.f

        self.rho = np.sum(f, axis=0)
        inverse_rho = 1./self.rho

        u = self.u
        v = self.v

        print inverse_rho.max()
        u = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])*inverse_rho
        v = (f[5]+f[2]+f[6]-f[7]-f[4]-f[8])*inverse_rho

        # Deal with boundary conditions...have to specify pressure
        lx = self.lx

        self.rho[0, :] = self.inlet_rho
        self.rho[lx, :] = self.outlet_rho
        u[0, :] = -1 + (f[0, 0, :]+f[2, 0, :]+f[4, 0, :]+2*(f[1, 0, :]+f[5, 0, :]+f[8, 0, :]))/self.inlet_rho
        u[lx, :] = -1 + (f[0, lx, :]+f[2, lx, :]+f[4, lx, :]+2*(f[1, lx, :]+f[5, lx, :]+f[8, lx, :]))/self.outlet_rho


    def move_bcs(self):
        """This is slow; cythonizing makes it fast."""

        cdef int lx = self.lx
        cdef int ly = self.ly
        cdef int i, j

        farr = self.f
        # EAST: constant pressure!
        farr[3, lx, :] = farr[1, lx, :] - (2./3.)*self.outlet_rho*self.u[lx,:]
        farr[7, lx, :] = farr[5, lx, :] + .5*(farr[2, lx, :] - farr[4, lx, :]) -(1./6.)*self.outlet_rho*self.u[lx, :]
        farr[6, lx, :] = farr[8, lx, :] - .5*(farr[2, lx, :] - farr[4, lx, :]) -(1./6.)*self.outlet_rho*self.u[lx, :]

        # WEST: constant pressure!
        farr[1, 0, :] = farr[3, 0, :] + (2./3.)*self.inlet_rho*self.u[0, :]
        farr[5, 0, :] = farr[7, 0, :] - .5*(farr[2, 0, :] - farr[4, 0, :]) + (1./6.)*self.inlet_rho*self.u[0, :]
        farr[8, 0, :] = farr[6, 0, :] + .5*(farr[2, 0, :] - farr[4, 0, :]) + (1./6.)*self.inlet_rho*self.u[0, :]

        cdef float[:, :, :] f = self.f

        with nogil:
            # NORTH solid
            for i in range(1, lx): # Bounce back
                f[4,i,ly] = f[2,i,ly-1]
                f[8,i,ly] = f[6,i+1,ly-1]
                f[7,i,ly] = f[5,i-1,ly-1]
            # SOUTH solid
            for i in range(1, lx):
                f[2,i,0] = f[4,i,1]
                f[6,i,0] = f[8,i-1,1]
                f[5,i,0] = f[7,i+1,1]

            # Corners bounce-back
            f[8,0,ly] = f[6,1,ly-1]
            f[5,0,0]  = f[7,1,1]
            f[7,lx,ly] = f[5,lx-1,ly-1]
            f[6,lx,0]  = f[8,lx-1,1]

    def move(self):
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

    def init_pop(self):
        feq = self.feq
        nx = self.nx
        ny = self.ny

        self.f = feq.copy()
        # We now slightly perturb f
        amplitude = .001
        perturb = (1. + amplitude*np.random.randn(nx, ny))
        self.f *= perturb

    def collide_particles(self):
        f = self.f
        feq = self.feq
        omega = self.omega

        self.f = f*(1.-omega)+omega*feq

    def run(self, num_iterations):
        for cur_iteration in range(num_iterations):
            self.move_bcs() # We have to udpate the boundary conditions first, or we are in trouble.
            self.move() # Move all jumpers
            self.update_hydro() # Update the hydrodynamic variables
            self.update_feq() # Update the equilibrium fields
            self.collide_particles() # Relax the nonequilibrium fields

