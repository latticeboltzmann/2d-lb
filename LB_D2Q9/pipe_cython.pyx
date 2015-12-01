#cython: profile=True
#cython: linetrace=True
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

from libc.stdio cimport printf
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

    def __init__(self, omega=.99, lx=400, ly=400, dr=1., dt = 1., deltaP=-.1):
        ### User input parameters
        self.lx = lx # Grid not including boundary in x
        self.ly = ly # Grid not including boundary in y

        self.omega = omega

        self.dr = dr
        self.dt = dt
        self.deltaP = deltaP

        ## Everything else
        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

        # Based on deltaP, set rho at the edges, as P = rho*cs^2, so rho=P/cs^2
        self.inlet_rho = 1.
        self.outlet_rho = self.deltaP/cs2 + self.inlet_rho # deltaP is negative!

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
        self.update_dimensionless_nums()

    def update_dimensionless_nums(self):
        self.viscosity = (self.dr**2/(3*self.dt))*(self.omega-0.5)

        # Get the reynolds number...based on max in the flow
        U = np.max(np.sqrt(self.u**2 + self.v**2))
        L = self.ly*self.dr # Diameter
        self.Re = U*L/self.viscosity

        # To get the mach number...
        self.Ma = (self.dr/(L*np.sqrt(3)))*(self.omega-.5)*self.Re


    def init_hydro(self):
        nx = self.nx
        ny = self.ny

        self.rho = np.ones((nx, ny), dtype=np.float32)
        self.rho[0, :] = self.inlet_rho
        self.rho[self.lx, :] = self.outlet_rho # Is there a shock in this case? We'll see...
        for i in range(self.rho.shape[0]):
            self.rho[i, :] = self.inlet_rho - i*(self.inlet_rho - self.outlet_rho)/float(self.rho.shape[0])

        self.u = .0*np.random.randn(nx, ny) # Fluctuations in the fluid
        self.v = .0*np.random.randn(nx, ny) # Fluctuations in the fluid


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

        rho = self.rho
        rho[:, :] = np.sum(f, axis=0)
        inverse_rho = 1./self.rho

        u = self.u
        v = self.v

        u[:, :] = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])*inverse_rho
        v[:, :] = (f[5]+f[2]+f[6]-f[7]-f[4]-f[8])*inverse_rho

        # Deal with boundary conditions...have to specify pressure
        lx = self.lx

        rho[0, :] = self.inlet_rho
        rho[lx, :] = self.outlet_rho
        # INLET
        u[0, :] = 1 - (f[0, 0, :]+f[2, 0, :]+f[4, 0, :]+2*(f[3, 0, :]+f[6, 0, :]+f[7, 0, :]))/self.inlet_rho

        # OUTLET
        u[lx, :] = -1 + (f[0, lx, :]+f[2, lx, :]+f[4, lx, :]+2*(f[1, lx, :]+f[5, lx, :]+f[8, lx, :]))/self.outlet_rho


    def move_bcs(self):
        """This is slow; cythonizing makes it fast."""

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
        amplitude = .00
        perturb = (1. + amplitude*np.random.randn(nx, ny))
        self.f *= perturb

    def collide_particles(self):
        f = self.f
        feq = self.feq
        omega = self.omega

        self.f[:, :, :] = f*(1.-omega)+omega*feq

    def run(self, num_iterations):
        for cur_iteration in range(num_iterations):
            self.move_bcs() # We have to udpate the boundary conditions first, or we are in trouble.
            self.move() # Move all jumpers
            self.update_hydro() # Update the hydrodynamic variables
            self.update_feq() # Update the equilibrium fields
            self.collide_particles() # Relax the nonequilibrium fields


class Pipe_Flow_Obstacles(Pipe_Flow):

    def __init__(self, *args, obstacle_mask=None, **kwargs):

        self.obstacle_mask = obstacle_mask
        self.obstacle_pixels = np.where(self.obstacle_mask)

        super(Pipe_Flow_Obstacles, self).__init__(*args, **kwargs)

    def init_hydro(self):
        super(Pipe_Flow_Obstacles, self).init_hydro()
        self.u[self.obstacle_mask] = 0
        self.v[self.obstacle_mask] = 0

    def update_hydro(self):
        super(Pipe_Flow_Obstacles, self).update_hydro()
        self.u[self.obstacle_mask] = 0
        self.v[self.obstacle_mask] = 0

    def move_bcs(self):
        Pipe_Flow.move_bcs(self)

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