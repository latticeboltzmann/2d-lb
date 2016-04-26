import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import ctypes as ct

# Required to draw obstacles
import skimage as ski
import skimage.draw

float_size = ct.sizeof(ct.c_float)

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
cs=1./np.sqrt(3)                         # Speed of sound on the lattice

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

class Diffusion(object):
    """
    Simulates pipe flow using the D2Q9 lattice. Generally used to verify that our simulations were working correctly.
    For usage, see the docs folder.
    """

    def __init__(self, Lx=1.0, Ly=1.0, D=1.0, Ro=0.1, time_prefactor = 1., N=50,
                 two_d_local_size=(32,32), three_d_local_size=(32,32,1), use_interop=False):
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
        :param two_d_local_size: A tuple of the local size to be used in 2d, i.e. (32, 32)
        :param three_d_local_size: A tuple of the local size to be used in 3d, i.e. (32, 32, 3)
        """

        # Physical units
        self.phys_Lx = Lx
        self.phys_Ly = Ly
        self.phys_D = D
        self.phys_Ro = Ro

        self.use_interop=use_interop

        # Get the characteristic length and time scales for the flow
        self.L = None # Characteristic length scale
        self.T = None # Characteristic time scale
        self.set_characteristic_length_time()
        print 'Characteristic L:', self.L
        print 'Characteristic T:', self.T

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = 1./N # How many squares characteristic length is broken into
        self.delta_t = time_prefactor * self.delta_x**2 # How many time iterations until the characteristic time, should be ~ \delta x^2

        self.ulb = self.delta_t/self.delta_x
        print 'u_lb:', self.ulb

        # Get omega from lb_viscosity
        # Note that lb_viscosity is basically constant as a function of grid size, as delta_t ~ delta_x**2.
        self.lb_D = self.delta_t/self.delta_x**2

        self.omega = (0.5 + 1/(cs**2 * self.delta_x**2))**-1. # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.

        # Initialize grid dimensions
        self.lx = None # Number of grid points in the x direction, ignoring the boundary
        self.ly = None # Number of grid points in the y direction, ignoring the boundary
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
        self.local_u = None
        self.local_v = None
        self.local_rho = None
        self.allocate_constants()

        ## Initialize hydrodynamic variables
        self.rho = None # The simulation's density field
        self.u = None # The simulation's velocity in the x direction (horizontal)
        self.v = None # The simulation's velocity in the y direction (vertical)
        self.init_hydro() # Create the hydrodynamic fields

        # Intitialize the underlying feq equilibrium field
        feq_host = np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F')
        self.feq = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=feq_host)

        self.update_feq() # Based on the hydrodynamic fields, create feq

        # Now initialize the nonequilibrium f
        f_host=np.zeros((self.nx, self.ny, NUM_JUMPERS), dtype=np.float32, order='F')
        # In order to stream in parallel without communication between workgroups, we need two buffers (as far as the
        # authors can see at least). f will be the usual field of hopping particles and f_streamed will be the field
        # after the particles have streamed.
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f_host)
        self.f_streamed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f_host)

        self.init_pop() # Based on feq, create the hopping non-equilibrium fields

    def set_characteristic_length_time(self):
        """
        Based on the input parameters, set the characteristic length and time scales. Required to make
        the simulation dimensionless. See http://www.latticeboltzmann.us/home/model-verification for more details.
        For pipe flow, L is the physical diameter of the pipe, and T is the time it takes the fluid moving at its
        theoretical maximum to to move a distance of L.
        """
        self.L = self.phys_Ro
        self.T = self.phys_Ro**2/self.phys_D

    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """

        self.lx = self.N*int(self.phys_Lx/self.L)
        self.ly = self.N*int(self.phys_Ly/self.L)

        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

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
        self.kernels = cl.Program(self.context, open(parent_dir + '/D2Q9_diffusion.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """

        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        self.local_u = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])
        self.local_v = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])
        self.local_rho = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])


    def init_hydro(self):
        """
        Based on the initial conditions, initialize the hydrodynamic fields, like density and velocity
        """

        nx = self.nx
        ny = self.ny

        # Only density in the innoculated region.
        rho_host = np.zeros((nx, ny), dtype=np.float32, order='F')

        x_center = self.N * (self.phys_Lx/2.) / self.L
        y_center = self.N * (self.phys_Ly/2.) / self.L

        circle = ski.draw.circle(x_center, y_center, self.N)
        rho_host[circle[0], circle[1]] = 1.0

        u_host = 0.0*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        u_host = u_host.astype(np.float32, order='F')
        v_host = 0.0*np.random.randn(nx, ny) # Fluctuations in the fluid; small
        v_host = v_host.astype(np.float32, order='F')

        # Transfer arrays to the device
        self.rho = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=rho_host)
        self.u = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=u_host)
        self.v = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_host)

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """
        self.kernels.update_feq_diffusion(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.feq,
                                self.u, self.v, self.rho,
                                self.local_u, self.local_v, self.local_rho,
                                self.w, self.cx, self.cy,
                                np.float32(cs), np.int32(self.nx), np.int32(self.ny)).wait()

    def init_pop(self):
        """Based on feq, create the initial population of jumpers."""

        nx = self.nx
        ny = self.ny

        # For simplicity, copy feq to the local host, where you can make a copy. There is probably a better way to do this.
        f = np.zeros((nx, ny, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.feq, is_blocking=True)

        # We now slightly perturb f
        amplitude = .001
        perturb = (1. + amplitude*np.random.randn(nx, ny, NUM_JUMPERS))
        f *= perturb

        # Now send f to the GPU
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

        # f_streamed will be the buffer that f moves into in parallel.
        self.f_streamed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        Implemented in OpenCL.
        """
        pass

    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """
        self.kernels.move(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f, self.f_streamed,
                                self.cx, self.cy,
                                np.int32(self.nx), np.int32(self.ny)).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        self.kernels.copy_buffer(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f_streamed, self.f,
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        """
        self.kernels.update_hydro_diffusion(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f, self.u, self.v, self.rho,
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def collide_particles(self):
        """
        Relax the nonequilibrium f fields towards their equilibrium feq. Depends on omega. Implemented in OpenCL.
        """
        self.kernels.collide_particles(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f, self.feq, np.float32(self.omega),
                                np.int32(self.nx), np.int32(self.ny)).wait()

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


    def get_fields(self):
        """
        :return: Returns a dictionary of all fields. Transfers data from the GPU to the CPU.
        """
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


# class Pipe_Flow_Cylinder(Pipe_Flow):
#     """
#     A subclass of the Pipe Flow class that simulates fluid flow around a cylinder. This class can also be "hacked"
#     in its current state to simulate flow around arbitrary obstacles. See
#     https://github.com/latticeboltzmann/2d-lb/blob/master/docs/cs205_movie.ipynb for an example of how to do so.
#     """
#
#     def set_characteristic_length_time(self):
#         """
#         Sets the characteristic length and time scale. For the cylinder, the characteristic length scale is
#         the cylinder radius. The characteristic time scale is the time it takes the fluid in the pipe moving at its
#         theoretical maximum to move over the cylinder.
#         """
#         self.L = self.phys_cylinder_radius
#         zeta = np.abs(self.phys_pressure_grad) / self.phys_rho
#         self.T = np.sqrt(self.phys_cylinder_radius / zeta)
#
#     def initialize_grid_dims(self):
#         """Initializes the grid, like the superclass, but also initializes an appropriate mask of the obstacle."""
#
#         self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
#         self.ly = int(np.ceil((self.phys_diameter / self.L)*self.N))
#
#         self.nx = self.lx + 1 # Total size of grid in x including boundary
#         self.ny = self.ly + 1 # Total size of grid in y including boundary
#
#         ## Initialize the obstacle mask
#         self.obstacle_mask_host = np.zeros((self.nx, self.ny), dtype=np.int32, order='F')
#
#         # Initialize the obstacle in the correct place
#         x_cylinder = self.N * self.phys_cylinder_center[0]/self.L
#         y_cylinder = self.N * self.phys_cylinder_center[1]/self.L
#
#         circle = ski.draw.circle(x_cylinder, y_cylinder, self.N)
#         self.obstacle_mask_host[circle[0], circle[1]] = 1
#
#
#     def __init__(self, cylinder_center = None, cylinder_radius=None, **kwargs):
#         """
#         :param cylinder_center: The center of the cylinder in physical units.
#         :param cylinder_radius: The raidus of the cylinder in physical units.
#         :param kwargs: All keyword arguments required to initialize the pipe-flow class.
#         """
#
#         assert (cylinder_center is not None) # If the cylinder does not have a center, this code will explode
#         assert (cylinder_radius is not None) # If there are no obstacles, this will definitely not run.
#
#         self.phys_cylinder_center = cylinder_center # Center of the cylinder in physical units
#         self.phys_cylinder_radius = cylinder_radius # Radius of the cylinder in physical units
#
#         self.obstacle_mask_host = None # A boolean mask of the location of the obstacle
#         self.obstacle_mask = None   # A buffer of the boolean mask of the obstacle
#         super(Pipe_Flow_Cylinder, self).__init__(**kwargs) # Initialize the superclass
#
#     def init_hydro(self):
#         """
#         Overrides the init_hydro method in Pipe_Flow.
#         """
#         super(Pipe_Flow_Cylinder, self).init_hydro()
#
#         # Now create the obstacle mask on the device
#         self.obstacle_mask = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
#                                        hostbuf=self.obstacle_mask_host)
#
#         # Based on where the obstacle mask is, set velocity to zero, as appropriate.
#         self.kernels.set_zero_velocity_in_obstacle(self.queue, self.two_d_global_size, self.two_d_local_size,
#                                                    self.obstacle_mask, self.u, self.v,
#                                                    np.int32(self.nx), np.int32(self.ny)).wait()
#
#     def move_bcs(self):
#         """
#         Overrides the move_bcs method in Pipe_Flow
#         """
#         super(Pipe_Flow_Cylinder, self).move_bcs()
#         # Now bounceback on the obstacle
#         self.kernels.bounceback_in_obstacle(self.queue, self.two_d_global_size, self.two_d_local_size,
#                                             self.obstacle_mask, self.f,
#                                             np.int32(self.nx), np.int32(self.ny)).wait()