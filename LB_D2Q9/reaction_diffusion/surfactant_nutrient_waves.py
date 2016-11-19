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


class Surfactant_Nutrient_Wave(object):
    """
    Everything is in dimensionless units. It's just easier.
    """

    def __init__(self, Lx=1.0, Ly=1.0, vc=1., lam=1., Dn = 1./4., R0 = 5.,
                 time_prefactor=1., N=50,
                 two_d_local_size=(32,32), three_d_local_size=(32,32,1), use_interop=False,
                 check_max_ulb=False, mach_tolerance=0.1):
        """
        :param N: Resolution of the simulation. As N increases, the simulation should become more accurate. N determines
                  how many grid points the characteristic length scale is discretized into
        :param time_prefactor: In order for a simulation to be accurate, in general, the dimensionless
                               space discretization delta_t ~ delta_x^2 (see http://wiki.palabos.org/_media/howtos:lbunits.pdf).
                               In our simulation, delta_t = time_prefactor * delta_x^2. delta_x is determined automatically
                               by N.
        :param two_d_local_size: A tuple of the local size to be used in 2d, i.e. (32, 32)
        :param three_d_local_size: A tuple of the local size to be used in 3d, i.e. (32, 32, 3)
        """

        # Physical units
        self.Lx = Lx
        self.Ly = Ly
        self.D = 1./4.
        self.Dn = Dn
        self.G = 1. # Growth rate is dimensionalized to one
        self.vc = vc # Characteristic velocity, deals with height vs. how concentration impacts surface tension
        self.lam = lam # Screening length
        self.R0 = R0 # Initial radius of the droplet

        self.num_populations = np.int32(2) # Population field and nutrient field. Could be extended to surfactant field someday as well
        self.pop_index = 0
        self.nut_index = 1

        # Interop with OpenGL?
        self.use_interop = use_interop
        self.check_max_ulb = check_max_ulb
        self.mach_tolerance = mach_tolerance

        # Get the characteristic length and time scales for the flow. Since this simulation is in dimensionless units
        # they should both be one!
        self.L = 1.0 # Fisher length
        self.T = 1.0 # Time in generations

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = 1./N # How many squares characteristic length is broken into
        self.delta_t = time_prefactor * self.delta_x**2 # How many time iterations until the characteristic time, should be ~ \delta x^2

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

        # Nutrient field
        self.lb_Dn = self.Dn * (self.delta_t / self.delta_x ** 2)  # Diffusion constant in LB units
        self.lb_Dn = np.float32(self.lb_Dn)

        self.omega_n = (.5 + self.lb_Dn / cs ** 2) ** -1.  # The relaxation time of the jumpers in the simulation
        self.omega_n = np.float32(self.omega_n)
        print 'omega_n', self.omega_n
        assert self.omega_n < 2.

        self.lb_G = np.float32(self.G * self.delta_t)

        # Initialize grid dimensions
        self.nx = None # Number of grid points in the x direction with the boundray
        self.ny = None # Number of grid points in the y direction with the boundary
        self.initialize_grid_dims()

        # Create global & local sizes appropriately
        self.two_d_local_size = two_d_local_size        # The local size to be used for 2-d workgroups
        self.three_d_local_size = three_d_local_size    # The local size to be used for 3-d workgroups

        self.two_d_global_size = get_divisible_global((self.nx, self.ny), self.two_d_local_size)
        self.three_d_global_size = get_divisible_global((self.nx, self.ny, 9), self.three_d_local_size)

        print '2d global:' , self.two_d_global_size
        print '2d local:' , self.two_d_local_size
        print '3d global:' , self.three_d_global_size
        print '3d local:' , self.three_d_local_size

        # Initialize the opencl environment
        self.context = None     # The pyOpenCL context
        self.queue = None       # The queue used to issue commands to the desired device
        self.kernels = None     # Compiled OpenCL kernels
        self.init_opencl()      # Initializes all items required to run OpenCL code

        # Allocate constants & local memory for opencl
        self.w = None
        self.cx = None
        self.cy = None
        self.allocate_constants()

        ## Initialize hydrodynamic variables and poisson solver
        self.rho = None # The simulation's density field
        self.u = None # The simulation's velocity in the x direction (horizontal)
        self.v = None # The simulation's velocity in the y direction (vertical)

        self.x_center = None
        self.y_center = None
        self.X_dim = None
        self.Y_dim = None

        self.poisson_solver = None

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
        self.kernels = cl.Program(self.context, open(file_dir + '/surfactant_nutrient_waves.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """
        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)


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
        # Population field
        rho_host[:, :, self.pop_index] = np.exp(-(self.X**2 + self.Y**2)/self.R0**2)

        # Nutrient field
        rho_host[:, :, self.nut_index] = 1.0 #- rho_host[:, :, self.pop_index]

        # Send to device
        self.rho = cl.array.to_device(self.queue, rho_host)

        #### VELOCITY ####

        # Create u and v fields. Necessary to copy onto due to complex type issues...
        u_host = np.zeros((nx, ny), dtype=np.float32, order='F')
        v_host = np.zeros((nx, ny), dtype=np.float32, order='F')

        self.u = cl.array.to_device(self.queue, u_host)
        self.v = cl.array.to_device(self.queue, v_host)

        # Initialize via poisson solver...
        density_field = rho_host[:, :, self.pop_index]
        self.poisson_solver = sp.Screened_Poisson(density_field, cl_context=self.context, cl_queue = self.queue,
                                                  lam=self.lam, dx=self.delta_x)
        self.poisson_solver.create_grad_fields()

        self.update_u_and_v()

    def redo_initial_condition(self, rho_field):
        """After you have specified your own IC"""
        rho_host = rho_field.astype(dtype=np.float32, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)

        self.update_u_and_v()
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
                                self.u.data,
                                self.v.data,
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
        self.kernels.move_periodic(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f.data, self.f_streamed.data,
                                self.cx, self.cy,
                                self.nx, self.ny, self.num_populations).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        cl.enqueue_copy(self.queue, self.f.data, self.f_streamed.data)

    def update_u_and_v(self):
        # Update the charge field for the poisson solver
        #self.poisson_solver.charge = self.rho.astype(np.complex64, queue=self.queue)
        density_view = self.rho[:, :, self.pop_index]

        cl.enqueue_copy(self.queue, self.poisson_solver.charge.data, density_view.astype(np.complex64).data)

        self.poisson_solver.solve_and_update_grad_fields()
        xgrad = self.poisson_solver.xgrad
        ygrad = self.poisson_solver.ygrad

        #TODO: THIS SEEMS TO BE THE ONLY WAY TO PREVENT WEIRD TRANSPOSITION ERRORS. TRY TO FIX IN THE FUTURE.
        # THERE SEEMS TO BE AN ERROR ASSOCIATED WITH .REAL, IT SEEMS TO TRANSPOSE ELEMENTS IN A WAY I DON'T
        # UNDERSTAND
        cl.enqueue_copy(self.queue, self.u.data, xgrad.real.data)
        cl.enqueue_copy(self.queue, self.v.data, ygrad.real.data)

        self.u *= -self.vc * (self.delta_t / self.delta_x)
        self.v *= -self.vc * (self.delta_t / self.delta_x)

        if self.check_max_ulb:
            max_ulb = cl.array.max((self.u**2 + self.v**2)**.5, queue=self.queue)

            if max_ulb > cs*self.mach_tolerance:
                print 'max_ulb is greater than cs/10! Ma=', max_ulb/cs

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        """
        self.kernels.update_hydro(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f.data, self.u.data, self.v.data, self.rho.data,
                                self.nx, self.ny, self.num_populations).wait()
        self.update_u_and_v()

    def collide_particles(self):
        """
        Relax the nonequilibrium f fields towards their equilibrium feq. Depends on omega. Implemented in OpenCL.
        """
        self.kernels.collide_particles(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f.data,
                                self.feq.data,
                                self.rho.data,
                                self.omega, self.omega_n,
                                self.lb_G,
                                self.w,
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

    # def get_nondim_fields(self):
    #     """
    #     :return: Returns a dictionary of the fields scaled so that they are in non-dimensional form.
    #     """
    #     fields = self.get_fields()
    #
    #     fields['u'] *= self.delta_x/self.delta_t
    #     fields['v'] *= self.delta_x/self.delta_t
    #
    #     return fields


# class Clumpy_Screened_Fisher_Wave(Screened_Fisher_Wave):
#
#     def __init__(self, g = 1.0, **kwargs):
#         self.psi = None
#         self.g = g
#         super(Clumpy_Screened_Fisher_Wave, self).__init__(**kwargs) # Initialize the superclass
#         self.psi = self.rho.copy()
#
#     def update_psi(self):
#         self.psi[...] = (self.rho