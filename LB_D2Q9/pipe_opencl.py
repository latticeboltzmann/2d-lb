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

        feq_host = np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F')
        self.feq = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=feq_host)

        self.update_feq()

        f_host=np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F') # initializing f
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f_host)
        self.f_streamed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f_host)

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
        self.kernels = cl.Program(self.context, open(file_dir + '/D2Q9.cl').read()).build(options='')

    # def update_dimensionless_nums(self):
    #     self.viscosity = (self.dr**2/(3*self.dt))*(self.omega-0.5)
    #
    #     # Get the reynolds number...based on max in the flow
    #     U = np.max(np.sqrt(self.u**2 + self.v**2))
    #     L = self.ly*self.dr # Diameter
    #     self.Re = U*L/self.viscosity
    #
    #     # To get the mach number...
    #     self.Ma = (self.dr/(L*np.sqrt(3)))*(self.omega-.5)*self.Re


    def init_hydro(self):
        nx = self.nx
        ny = self.ny

        # Initialize arrays on the host
        rho_host = np.ones((nx, ny), dtype=np.float32, order='F')
        rho_host[0, :] = self.inlet_rho
        rho_host[self.lx, :] = self.outlet_rho # Is there a shock in this case? We'll see...
        for i in range(rho_host.shape[0]):
            rho_host[i, :] = self.inlet_rho - i*(self.inlet_rho - self.outlet_rho)/float(rho_host.shape[0])

        u_host = .0*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        u_host = u_host.astype(np.float32, order='F')
        v_host = .0*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        v_host = v_host.astype(np.float32, order='F')

        # Transfer arrays to the device
        self.rho = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=rho_host)
        self.u = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=u_host)
        self.v = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_host)

    def move_bcs(self):
        self.kernels.move_bcs(self.queue, (self.nx, self.ny), None,
                                self.f, self.u,
                                np.float32(self.inlet_rho), np.float32(self.outlet_rho),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def move(self):
        self.kernels.move(self.queue, (self.nx, self.ny, NUM_JUMPERS), None,
                                self.f, self.f_streamed,
                                np.int32(self.nx), np.int32(self.ny)).wait()

        # Replace f with f_streamed
        self.f, self.f_streamed = self.f_streamed, self.f


    def init_pop(self):
        nx = self.nx
        ny = self.ny

        # For simplicity, copy feq to the local host, where you can make a copy
        f = np.zeros((nx, ny, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.feq, is_blocking=True)

        # We now slightly perturb f
        amplitude = .00
        perturb = (1. + amplitude*np.random.randn(nx, ny, NUM_JUMPERS))
        f *= perturb

        # Now send f to the GPU
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

        # Create a new buffer
        self.f_streamed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

    def update_hydro(self):
        self.kernels.update_hydro(self.queue, (self.nx, self.ny), None,
                                self.f, self.u, self.v, self.rho,
                                np.float32(self.inlet_rho), np.float32(self.outlet_rho),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def update_feq(self):
        self.kernels.update_feq(self.queue, (self.nx, self.ny, NUM_JUMPERS), None,
                                self.feq, self.u, self.v, self.rho,
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def collide_particles(self):
        self.kernels.collide_particles(self.queue, (self.nx, self.ny, NUM_JUMPERS), None,
                                self.f, self.feq, np.float32(self.omega),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def run(self, num_iterations):
        for cur_iteration in range(num_iterations):
            self.move_bcs() # We have to udpate the boundary conditions first, or we are in trouble.
            self.move() # Move all jumpers
            # Update the hydrodynamic variables
            self.update_hydro()
            # Update the equilibrium fields
            self.update_feq()
            # Relax the nonequilibrium fields
            self.collide_particles()

    def get_fields_on_cpu(self):
        f = np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.f, is_blocking=True)

        feq = np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, feq, self.feq, is_blocking=True)

        u = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, u, self.u, is_blocking=True)

        v = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, v, self.v, is_blocking=True)

        rho = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, rho, self.rho, is_blocking=True)

        results={}
        results['f'] = f
        results['u'] = u
        results['v'] = v
        results['rho'] = rho
        results['feq'] = feq
        return results

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