import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
from LB_D2Q9.spectral_poisson import screened_poisson as sp
import matplotlib.pyplot as plt

# Required to draw obstacles
import skimage as ski
import skimage.draw

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)
parent_dir = os.path.dirname(file_dir)

# Required for allocating local memory
float_size = ct.sizeof(ct.c_float)

##########################
##### D2Q9 parameters ####
##########################
w=np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,
            1./36.,1./36.,1./36.], order='F', dtype=np.float32)      # weights for directions
cx=np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=np.int32)     # direction vector for the x direction
cy=np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=np.int32)     # direction vector for the y direction
cs=np.float32(1./np.sqrt(3))                         # Speed of sound on the lattice

w0 = 4./9.                              # Weight of stationary jumpers
w1 = 1./9.                              # Weight of horizontal and vertical jumpers
w2 = 1./36.                             # Weight of diagonal jumpers

NUM_JUMPERS = 9                         # Number of jumpers for the D2Q9 lattice: 9


def get_divisible_global(global_size, local_size):
    """
    Given a desired global size and a specified local size, return the smallest global
    size that the local size fits into. Required when specifying arbitrary local
    workgroup sizes.

    :param global_size: A tuple of the global size, i.e. (x, y, z)
    :param local_size:  A tuple of the local size, i.e. (lx, ly, lz)
    :return: The smallest global size that the local size fits into.
    """
    new_size = []
    for cur_global, cur_local in zip(global_size, local_size):
        remainder = cur_global % cur_local
        if remainder == 0:
            new_size.append(cur_global)
        else:
            new_size.append(cur_global + cur_local - remainder)
    return tuple(new_size)


class Rocket_Yeast_Forces_Only(object):
    """
    Everything is in dimensionless units. It's just easier.
    """

    def __init__(self, Lx=1.0, Ly=1.0, R0 = 5., epsilon=1., Dc = 1./4., Gc = 2.0,
                 rho_o=1.0, G_chen=-1.0,
                 time_prefactor=1., N=10,
                 two_d_local_size=(32,32), use_interop=False,
                 check_max_ulb=False, mach_tolerance=0.1):
        """
        :param N: Resolution of the simulation. As N increases, the simulation should become more accurate. N determines
                  how many grid points the characteristic length scale is discretized into
        :param time_prefactor: In order for a simulation to be accurate, in general, the dimensionless
                               space discretization delta_t ~ delta_x^2 (see http://wiki.palabos.org/_media/howtos:lbunits.pdf).
                               In our simulation, delta_t = time_prefactor * delta_x^2. delta_x is determined automatically
                               by N.
        :param two_d_local_size: A tuple of the local size to be used in 2d, i.e. (32, 32)
        """

        # Physical units
        self.Lx = Lx
        self.Ly = Ly
        # Population growth and diffusion constant are set by our non-dimensionalization
        self.D = 1./4.
        self.G = 1.

        # Surfactant diffusion constant and "growth rate" are set by the user
        self.Dc = (1./4.)*np.float32(Dc)
        self.Gc = np.float32(Gc)
        self.epsilon = np.float32(epsilon) # Characteristic velocity coupling constant, deals with height vs. how concentration impacts surface tension

        self.R0 = R0 # Initial radius of the droplet

        # For clumpiness, self-attraction of yeast
        self.rho_o = np.float32(rho_o)
        self.G_chen = np.float32(G_chen)

        # Book-keeping
        self.num_populations = np.int32(2) # Population and surfactant
        self.pop_index = np.int32(0)
        self.surf_index = np.int32(1)

        # Get the characteristic length and time scales for the flow. Since this simulation is in dimensionless units
        # they should both be one!
        self.L = 1.0 # Fisher length
        self.T = 1.0 # Time in generations

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = np.float32(1./N) # How many squares characteristic length is broken into
        self.delta_t = np.float32(time_prefactor * self.delta_x**2) # How many time iterations until the characteristic time, should be ~ \delta x^2

        # Characteristic LB speed corresponding to dimensionless speed of 1. Must be MUCH smaller than cs = .57 or so.
        self.ulb = self.delta_t/self.delta_x
        print 'u_lb:', self.ulb

        # Population field
        self.lb_D = self.D * (self.delta_t / self.delta_x ** 2) # Diffusion constant in LB units
        self.lb_D = np.float32(self.lb_D)
        self.omega = (.5 + self.lb_D / cs ** 2) ** -1.  # The relaxation time of the jumpers in the simulation
        self.omega = np.float32(self.omega)
        print 'omega', self.omega
        assert self.omega < 2.

        self.lb_G = np.float32(self.G * self.delta_t)

        # Surfactant Field
        self.lb_Dc = self.Dc * (self.delta_t / self.delta_x ** 2)  # Diffusion constant in LB units
        self.lb_Dc = np.float32(self.lb_Dc)

        self.omega_c = (.5 + self.lb_Dc / cs ** 2) ** -1.  # The relaxation time of the jumpers in the simulation
        self.omega_c = np.float32(self.omega_c)
        print 'omega_s', self.omega_c
        assert self.omega_c < 2.

        self.lb_Gc = np.float32(self.Gc * self.delta_t)

        # Initialize grid dimensions
        self.nx = None # Number of grid points in the x direction with the boundray
        self.ny = None # Number of grid points in the y direction with the boundary
        self.initialize_grid_dims()

        # Create global & local sizes appropriately
        self.two_d_local_size = two_d_local_size        # The local size to be used for 2-d workgroups
        self.two_d_global_size = get_divisible_global((self.nx, self.ny), self.two_d_local_size)

        print '2d global:' , self.two_d_global_size
        print '2d local:' , self.two_d_local_size

        # Initialize the opencl environment
        self.context = None     # The pyOpenCL context
        self.queue = None       # The queue used to issue commands to the desired device
        self.kernels = None     # Compiled OpenCL kernels
        self.use_interop = use_interop
        self.init_opencl()      # Initializes all items required to run OpenCL code

        # Allocate constants & local memory for opencl
        self.w = None
        self.cx = None
        self.cy = None

        self.halo = None
        self.buf_nx = None
        self.buf_ny = None
        self.psi_local = None

        self.allocate_constants()

        ## Initialize hydrodynamic variables & Shan-chen variables
        self.rho = None # The simulation's density field
        self.u = None # Velocity in the x direction
        self.v = None # Velocity in the y direction
        self.m = None # Mobility field

        self.surface_force_x = None # The simulation's velocity in the x direction (horizontal)
        self.surface_force_y = None # The simulation's velocity in the y direction (vertical)

        self.psi = None # Accounts for vanderwalls forces between strains
        self.pseudo_force_x = None
        self.pseudo_force_y = None

        self.x_center = None
        self.y_center = None
        self.X_dim = None
        self.Y_dim = None

        self.init_hydro() # Create the hydrodynamic fields, and also intialize the poisson solver

        # Intitialize the underlying feq equilibrium field
        feq_host = np.zeros((self.nx, self.ny, self.num_populations, NUM_JUMPERS), dtype=np.float32, order='F')
        self.feq = cl.array.to_device(self.queue, feq_host)

        self.update_feq() # Based on the hydrodynamic fields, create feq

        # Now initialize the nonequilibrium f
        # In order to stream in parallel without communication between workgroups, we need two buffers (as far as the
        # authors can see at least). f will be the usual field of hopping particles and f_temporary will be the field
        # after the particles have streamed.

        self.f = None
        self.f_streamed = None

        self.init_pop() # Based on feq, create the hopping non-equilibrium fields


    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """
        self.nx = np.int32(np.round(self.N*self.Lx))
        self.ny = np.int32(np.round(self.N*self.Ly))

        print 'nx:' , self.nx
        print 'ny:', self.ny

    def init_opencl(self):
        """
        Initializes the base items needed to run OpenCL code.
        """

        # Startup script shamelessly taken from CS205 homework
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
        if not self.use_interop:
            self.context = cl.Context(devices)
        else:
            self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, platforms[0])]
                                                 + cl.tools.get_gl_sharing_context_properties(),
                                      devices= devices)
        print 'This context is associated with ', len(self.context.devices), 'devices'

        # Create a simple queue
        self.queue = cl.CommandQueue(self.context, self.context.devices[0],
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        # Compile our OpenCL code
        self.kernels = cl.Program(self.context, open(file_dir + '/rocket_yeast_forces_only.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """
        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        # Allocate local memory for the clumpiness
        self.halo = np.int32(1) # As we are doing D2Q9, we have a halo of one
        self.buf_nx = np.int32(self.two_d_local_size[0] + 2 * self.halo)
        self.buf_ny = np.int32(self.two_d_local_size[1] + 2 * self.halo)
        self.psi_local = cl.LocalMemory(float_size * self.buf_nx * self.buf_ny)


    def init_hydro(self):
        """
        Based on the initial conditions, initialize the hydrodynamic fields, like density and velocity.
        This involves creating the poisson solver and solving for the velocity fields.
        """

        nx = self.nx
        ny = self.ny

        #### COORDINATE SYSTEM: FOR CHECKING SIMULATIONS ####
        self.x_center = nx/2
        self.y_center = ny/2

        # Now initialize the gaussian
        xvalues = np.arange(nx)
        yvalues = np.arange(ny)
        Y, X = np.meshgrid(yvalues, xvalues)
        X = X.astype(np.float)
        Y = Y.astype(np.float)

        deltaX = X - self.x_center
        deltaY = Y - self.y_center

        # Convert to dimensionless coordinates
        self.X = deltaX / self.N
        self.Y = deltaY / self.N

        #### DENSITY #####
        rho_host = np.zeros((nx, ny, self.num_populations), dtype=np.float32, order='F')

        ## Population field
        rho_host[:, :, self.pop_index] = 1.0*np.exp(-(self.X**2 + self.Y**2)/self.R0**2)*(1 + .05*np.random.randn(nx, ny))

        ## Surfactant field
        rho_host[:, :, self.surf_index] = 0.0 # No surfactant initially

        # Send to device
        self.rho = cl.array.to_device(self.queue, rho_host)

        ### Mobility ###
        m = np.zeros((nx, ny), dtype=np.float32, order='F')
        self.m = cl.array.to_device(self.queue, m)

        ### Velocity ###
        u = np.zeros((nx, ny), dtype=np.float32, order='F')
        v = np.zeros((nx, ny), dtype=np.float32, order='F')

        self.u = cl.array.to_device(self.queue, u)
        self.v = cl.array.to_device(self.queue, v)

        #### FORCES ####

        # Surface forces
        surface_force_x = np.zeros((nx, ny), dtype=np.float32, order='F')
        surface_force_y = np.zeros((nx, ny), dtype=np.float32, order='F')

        self.surface_force_x = cl.array.to_device(self.queue, surface_force_x)
        self.surface_force_y = cl.array.to_device(self.queue, surface_force_y)

        # Van-der-waals forces
        psi_host = np.zeros((self.ny, self.ny), dtype=np.float32, order='F')
        self.psi = cl.array.to_device(self.queue, psi_host)

        pseudo_force_host = np.zeros((self.ny, self.ny), dtype=np.float32, order='F')
        self.pseudo_force_x = cl.array.to_device(self.queue, pseudo_force_host)
        self.pseudo_force_y = self.pseudo_force_x.copy()

        # Now initialize all forces present
        self.update_forces()

    def redo_initial_condition(self, rho_field):
        """After you have specified your own IC"""
        rho_host = rho_field.astype(dtype=np.float32, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)

        self.update_forces()

        self.update_feq()  # Based on the hydrodynamic fields, create feq
        self.init_pop()  # Based on feq, create the hopping non-equilibrium fields

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """
        self.kernels.update_feq(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.feq.data,
                                self.rho.data,
                                self.w, self.cx, self.cy, cs,
                                self.nx, self.ny, self.num_populations).wait()

    def init_pop(self, amplitude=0):
        """Based on feq, create the initial population of jumpers."""

        nx = self.nx
        ny = self.ny

        # For simplicity, copy feq to the local host, where you can make a copy. There is probably a better way to do this.
        f_host = self.feq.get()

        # We now slightly perturb f. This is actually dangerous, as concentration can grow exponentially fast
        # from sall fluctuations. Sooo...be careful.
        perturb = (1. + amplitude*np.random.randn(nx, ny, self.num_populations, NUM_JUMPERS))
        f_host *= perturb

        # Now send f to the GPU
        self.f = cl.array.to_device(self.queue, f_host)

        # f_temporary will be the buffer that f moves into in parallel.
        self.f_streamed = self.f.copy()

    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        Implemented in OpenCL.
        """
        pass # Implemented in move_periodic in this case...it's just easier

    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """
        self.kernels.move_periodic(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f.data, self.f_streamed.data,
                                self.cx, self.cy,
                                self.nx, self.ny, self.num_populations).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        cl.enqueue_copy(self.queue, self.f.data, self.f_streamed.data)

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        """
        self.kernels.update_hydro(self.queue, self.two_d_global_size, self.two_d_local_size,
                                  self.f.data,
                                  self.mom_x, self.mom_y, self.rho.data,
                                  self.nx, self.ny, self.num_populations).wait()
        self.update_forces()

    def update_forces(self):

        ### Forces due to fluid shear ###
        self.kernels.update_surface_forces(self.queue, self.two_d_global_size, self.two_d_local_size,
                                    self.rho.data, self.surface_force_x.data, self.surface_force_y.data,
                                    self.delta_x,
                                    self.surf_index, cs, self.epsilon,
                                    self.cx, self.cy, self.w,
                                    self.psi_local,
                                    self.nx, self.ny,
                                    self.buf_nx, self.buf_ny, self.halo).wait()

        ### Forces due to van-der waals interactions ###
        self.kernels.update_psi(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.psi.data,
                                self.rho.data,
                                self.rho_o,
                                self.nx, self.ny, self.pop_index).wait()

        # self.kernels.update_psi_sticky_repulsive(self.queue, self.two_d_global_size, self.two_d_local_size,
        #                         self.psi.data,
        #                         self.rho.data,
        #                         self.rho_o,
        #                         self.nx, self.ny, self.pop_index).wait()

        self.kernels.update_pseudo_force(self.queue, self.two_d_global_size, self.two_d_local_size,
                                         self.psi.data,
                                         self.pseudo_force_x.data,
                                         self.pseudo_force_y.data,
                                         self.G_chen,
                                         cs,
                                         self.cx,
                                         self.cy,
                                         self.w,
                                         self.psi_local,
                                         self.nx, self.ny, self.buf_nx, self.buf_ny,
                                         self.halo).wait()

    def collide_particles(self):
        self.kernels.collide_particles(self.queue, self.two_d_global_size, self.two_d_local_size,
                                       self.f.data,
                                       self.feq.data,
                                       self.rho.data,
                                       self.omega, self.omega_c,
                                       self.lb_G, self.lb_Gc,
                                       self.pseudo_force_x.data, self.pseudo_force_y.data,
                                       self.surface_force_x.data, self.surface_force_y.data,
                                       self.w, self.cx, self.cy, cs,
                                       self.nx, self.ny, self.num_populations).wait()


    def run(self, num_iterations):
        """
        Run the simulation for num_iterations. Be aware that the same number of iterations does not correspond
        to the same non-dimensional time passing, as delta_t, the time discretization, will change depending on
        your resolution.

        :param num_iterations: The number of iterations to run
        """
        for cur_iteration in range(num_iterations):
            self.move() # Move all jumpers
            self.move_bcs() # Our BC's rely on streaming before applying the BC, actually

            self.update_hydro() # Update the hydrodynamic variables
            self.update_feq() # Update the equilibrium fields
            self.collide_particles() # Relax the nonequilibrium fields.