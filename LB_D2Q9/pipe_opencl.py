import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import ctypes as ct

float_size = ct.sizeof(ct.c_float)

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)

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

        self.omega = np.float32(omega)

        self.dr = np.float32(dr)
        self.dt = np.float32(dt)
        self.deltaP = np.float32(deltaP)

        ## Everything else
        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

        # Based on deltaP, set rho at the edges, as P = rho*cs^2, so rho=P/cs^2
        self.inlet_rho = 1.
        self.outlet_rho = self.deltaP/cs2 + self.inlet_rho # deltaP is negative!

        # Initialize the opencl environment
        self.context = None
        self.queue = None
        self.kernels = None
        self.init_opencl()

        ## Initialize hydrodynamic variables
        self.rho = None # Density
        self.u = None # Horizontal flow
        self.v = None # Vertical flow
        self.init_hydro()

        # Intitialize the underlying probablistic fields
        f_host=np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32) # initializing f
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*f_host.size)

        feq_host = np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32)
        self.feq = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*feq_host.size)

        print 'Starting kernel...'
        self.kernels.update_feq(self.queue, (self.nx, self.ny, NUM_JUMPERS), None,
                                self.feq, self.u, self.v, self.rho,
                                np.int32(self.nx), np.int32(self.ny)).wait()
        print 'Done!'
        self.init_pop()

        # Based on initial parameters, determine dimensionless numbers
        #self.viscosity = None
        #self.Re = None
        #self.Ma = None
        #self.update_dimensionless_nums()

    def init_opencl(self):
        platforms = cl.get_platforms()
        print 'The platforms detected are:'
        print '---------------------------'
        for platform in platforms:
            print platform.name, platform.vendor, 'version:', platform.version

        # List devices in each platform
        for platform in platforms:
            print 'The devices detected on platform', platform.name, 'are:'
            print '---------------------------'
            for device in platform.get_devices():
                print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                print 'Maximum work group size', device.max_work_group_size
                print 'Maximum work item dimensions', device.max_work_item_dimensions
                print 'Maximum work item size', device.max_work_item_sizes
                print '---------------------------'

        # Create a context with all the devices
        devices = platforms[0].get_devices()
        self.context = cl.Context(devices)
        print 'This context is associated with ', len(self.context.devices), 'devices'
        self.queue = cl.CommandQueue(self.context, self.context.devices[0],
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.kernels = cl.Program(self.context, open(file_dir + '/D2Q9.cl').read()).build(options='-g -cl-opt-disable')

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

        # Initialize arrays on the host
        rho_host = np.ones((nx, ny), dtype=np.float32)
        rho_host[0, :] = self.inlet_rho
        rho_host[self.lx, :] = self.outlet_rho # Is there a shock in this case? We'll see...
        for i in range(rho_host.shape[0]):
            rho_host[i, :] = self.inlet_rho - i*(self.inlet_rho - self.outlet_rho)/float(rho_host.shape[0])

        u_host = .0001*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        u_host = u_host.astype(np.float32)
        v_host = .0001*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        v_host = v_host.astype(np.float32)

        # Transfer arrays to the device
        self.rho = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*rho_host.size)
        self.u = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*u_host.size)
        self.v = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*v_host.size)

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

        lx = self.lx
        ly = self.ly

        farr = self.f

        # INLET: constant pressure!
        farr[1, 0, 1:ly] = farr[3, 0, 1:ly] + (2./3.)*self.inlet_rho*self.u[0, 1:ly]
        farr[5, 0, 1:ly] = -.5*farr[2,0,1:ly]+.5*farr[4, 0, 1:ly]+farr[7, 0, 1:ly] + (1./6.)*self.u[0, 1:ly]*self.inlet_rho
        farr[8, 0, 1:ly] = .5*farr[2,0,1:ly]-.5*farr[4, 0, 1:ly]+farr[6, 0, 1:ly] + (1./6.)*self.u[0, 1:ly]*self.inlet_rho

        # OUTLET: constant pressure!
        farr[3, lx, 1:ly] = farr[1, lx, 1:ly] - (2./3.)*self.outlet_rho*self.u[lx,1:ly]
        farr[6, lx, 1:ly] = -.5*farr[2,lx,1:ly]+.5*farr[4,lx,1:ly]+farr[8,lx,1:ly]-(1./6.)*self.u[lx,1:ly]*self.outlet_rho
        farr[7, lx, 1:ly] = .5*farr[2,lx,1:ly]-.5*farr[4,lx,1:ly]+farr[5,lx,1:ly]-(1./6.)*self.u[lx,1:ly]*self.outlet_rho

        f = self.f
        inlet_rho = self.inlet_rho
        outlet_rho = self.outlet_rho

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
        f[8, 0, ly] = f[6, 0, ly]
        f[5, 0, ly] = .5*(-f[0,0,ly]-2*f[2,0,ly]-2*f[3,0,ly]-2*f[6,0,ly]+inlet_rho)
        f[7, 0, ly] = .5*(-f[0,0,ly]-2*f[2,0,ly]-2*f[3,0,ly]-2*f[6,0,ly]+inlet_rho)

        # BOTTOM OUTLET
        f[3, lx, 0] = f[1, lx, 0]
        f[2, lx, 0] = f[4, lx, 0]
        f[6, lx, 0] = f[8, lx, 0]
        f[5, lx, 0] = .5*(-f[0,lx,0]-2*f[1,lx,0]-2*f[4,lx,0]-2*f[8,lx,0]+outlet_rho)
        f[8, lx, 0] = .5*(-f[0,lx,0]-2*f[1,lx,0]-2*f[4,lx,0]-2*f[8,lx,0]+outlet_rho)

        # TOP OUTLET
        f[3, lx, ly] = f[1, lx, ly]
        f[4, lx, ly] = f[2, lx, ly]
        f[7, lx, ly] = f[5, lx, ly]
        f[6, lx, ly] = .5*(-f[0,lx,ly]-2*f[1,ly,ly]-2*f[2,lx,ly]-2*f[5,lx,ly]+outlet_rho)
        f[8, lx, ly] = .5*(-f[0,lx,ly]-2*f[1,ly,ly]-2*f[2,lx,ly]-2*f[5,lx,ly]+outlet_rho)

    def move(self):
        f = self.f
        lx = self.lx
        ly = self.ly

        # This can't be parallelized without making a copy...order of loops is super important!
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

        # For simplicity, copy feq to the local host, where you can make a copy
        f = np.zeros((nx, ny, NUM_JUMPERS), dtype=np.float32)
        cl.enqueue_copy(self.queue, self.feq, f, is_blocking=True)

        f = f.copy() # Make sure there is no problem
        # We now slightly perturb f
        amplitude = .001
        perturb = (1. + amplitude*np.random.randn(nx, ny, NUM_JUMPERS))
        f *= perturb

        # Now send f to the GPU
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, float_size*f.size)

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

    def __init__(self, obstacle_mask=None, **kwargs):
        self.obstacle_mask = obstacle_mask
        self.obstacle_pixels = np.where(self.obstacle_mask)

        super(Pipe_Flow_Obstacles, self).__init__(**kwargs)

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
        x_list = self.obstacle_pixels[0]
        y_list = self.obstacle_pixels[1]
        num_pixels = y_list.shape[0]

        f = self.f


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