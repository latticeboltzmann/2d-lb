import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
import matplotlib.pyplot as plt

# Required to draw obstacles
import skimage as ski
import skimage.draw

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)
parent_dir = os.path.dirname(file_dir)

# Required for allocating local memory
num_size = ct.sizeof(ct.c_double)

num_type = np.double
int_type = np.int32


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

class Pourous_Media(object):

    def __init__(self, sim, field_index, nu_e = 1.0, epsilon = 1.0, nu_fluid=1.0, K=1.0, Fe=1.0,
                 bc='periodic'):

        self.sim = sim # TODO: MAKE THIS A WEAKREF

        self.field_index = int_type(field_index)

        self.nu_e = num_type(nu_e)
        self.epsilon = num_type(epsilon)
        self.nu_fluid = num_type(nu_fluid)
        self.K = num_type(K)
        self.Fe = num_type(Fe)
        self.bc = bc

        # Determine the viscosity
        self.lb_nu_e = self.nu_e * (sim.delta_t / sim.delta_x ** 2)
        self.tau = num_type(.5 + self.lb_nu_e / (sim.cs**2))
        print 'tau', self.tau
        self.omega = num_type(self.tau ** -1.)  # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.

        # Total force INCLUDING drag forces & body force G
        self.Fx = None
        self.Fy = None


    def initialize(self, u_arr, v_arr, rho_arr, f_amp = 0.0):
        """
        User passes in the u field. As density is fixed at a constant (incompressibility), we solve for the appropriate
        distribution functions.
        """

        #### VELOCITY ####
        u_host = self.sim.u.get()
        v_host = self.sim.v.get()

        u_host[:, :, self.field_index] = u_arr
        v_host[:, :, self.field_index] = v_arr

        self.sim.u = cl.array.to_device(self.sim.queue, u_host)
        self.sim.v = cl.array.to_device(self.sim.queue, v_host)

        #### DENSITY #####
        rho_host = self.sim.rho.get()

        rho_host[:, :, self.field_index] = rho_arr
        self.sim.rho = cl.array.to_device(self.sim.queue, rho_host)

        #### TOTAL FORCE, including internal drag ####
        Fx_host = np.zeros((self.sim.nx, self.sim.ny), dtype=num_type, order='F')
        Fy_host = np.zeros((self.sim.nx, self.sim.ny), dtype=num_type, order='F')

        self.Fx = cl.array.to_device(self.sim.queue, Fx_host)
        self.Fy = cl.array.to_device(self.sim.queue, Fy_host)

        self.update_forces()

        #### UPDATE HOPPERS ####
        self.update_feq() # Based on the hydrodynamic fields, create feq

        # Now initialize the nonequilibrium f
        # In order to stream in parallel without communication between workgroups, we need two buffers (as far as the
        # authors can see at least). f will be the usual field of hopping particles and f_temporary will be the field
        # after the particles have streamed.

        self.init_pop(amplitude=f_amp) # Based on feq, create the hopping non-equilibrium fields


    def init_pop(self, amplitude=0.001):
        """Based on feq, create the initial population of jumpers."""

        nx = self.sim.nx
        ny = self.sim.ny

        # For simplicity, copy feq to the local host, where you can make a copy. There is probably a better way to do this.
        f_host = self.sim.feq.get()
        cur_f = f_host[:, :, self.field_index, :]

        # We now slightly perturb f. This is actually dangerous, as concentration can grow exponentially fast
        # from sall fluctuations. Sooo...be careful.
        perturb = (1. + amplitude * np.random.randn(nx, ny, self.sim.num_jumpers))
        cur_f *= perturb

        # Now send f to the GPU
        f_host[:, :, self.field_index, :] = cur_f
        self.sim.f = cl.array.to_device(self.sim.queue, f_host)

    def update_forces(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_forces_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.u.data, sim.v.data,
            self.Fx.data, self.Fy.data,
            sim.Gx.data, sim.Gy.data,
            self.epsilon, self.nu_fluid, self.Fe, self.K,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations
        ).wait()

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_feq_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.feq.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            self.epsilon,
            sim.w, sim.cx, sim.cy, sim.cs,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers).wait()

    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        Implemented in OpenCL.
        """

        sim = self.sim

        if self.bc is 'periodic':
            pass # Implemented in move_periodic in this case...it's just easier
        elif self.bc is 'zero_gradient':
            self.sim.kernels.move_open_bcs(
                sim.queue, sim.two_d_global_size, sim.two_d_local_size,
                sim.f.data,
                sim.nx, sim.ny,
                self.field_index, sim.num_populations,
                sim.num_jumpers).wait()
        else:
            raise ValueError('unknown bc...')


    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """

        sim = self.sim

        if self.bc is 'periodic':
            self.sim.kernels.move_periodic(
                sim.queue, sim.two_d_global_size, sim.two_d_local_size,
                sim.f.data, sim.f_streamed.data,
                sim.cx, sim.cy,
                sim.nx, sim.ny,
                self.field_index, sim.num_populations, sim.num_jumpers
            ).wait()
        elif self.bc is 'zero_gradient':
            self.sim.kernels.move(
                sim.queue, sim.two_d_global_size, sim.two_d_local_size,
                sim.f.data, sim.f_streamed.data,
                sim.cx, sim.cy,
                sim.nx, sim.ny,
                self.field_index, sim.num_populations, sim.num_jumpers
            ).wait()
        else:
            raise ValueError('unknown bc...')

        # Copy the streamed buffer into f so that it is correctly updated.
        self.sim.kernels.copy_streamed_onto_f(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f_streamed.data, sim.f.data,
            sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations, sim.num_jumpers).wait()

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        Requires u_prime to have been updated first!
        """

        sim = self.sim

        sim.kernels.update_hydro_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.rho.data,
            sim.u_prime.data, sim.v_prime.data,
            sim.u.data, sim.v.data,
            sim.Gx.data, sim.Gy.data,
            self.epsilon, self.nu_fluid, self.Fe, self.K,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers, self.sim.delta_t
        ).wait()

        if sim.check_max_ulb:
            max_ulb = cl.array.max((sim.u**2 + sim.v**2)**.5, queue=self.queue)

            if max_ulb > sim.cs*sim.mach_tolerance:
                print 'max_ulb is greater than cs/10! Ma=', max_ulb/sim.cs

    def collide_particles(self):
        sim = self.sim

        self.sim.kernels.collide_particles_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.feq.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            self.Fx.data, self.Fy.data,
            self.epsilon, self.omega,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers, sim.delta_t, sim.cs
        ).wait()

class Simulation_Runner(object):
    """
    Everything is in dimensionless units. It's just easier.
    """

    def __init__(self, Lx=1.0, Ly=1.0,
                 time_prefactor=1., N=10, num_populations=1,
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

        # Dimensionless units
        self.Lx = Lx
        self.Ly = Ly

        # Book-keeping
        self.num_populations = int_type(num_populations)

        self.check_max_ulb = check_max_ulb
        self.mach_tolerance = mach_tolerance

        # Get the characteristic length and time scales for the flow.
        self.L = 1.0 # mm
        self.T = 1.0 # Time in generations

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = num_type(1./N) # How many squares characteristic length is broken into
        self.delta_t = num_type(time_prefactor * self.delta_x**2) # How many time iterations until the characteristic time, should be ~ \delta x^2

        # Characteristic LB speed corresponding to dimensionless speed of 1. Must be MUCH smaller than cs = .57 or so.
        self.ulb = self.delta_t/self.delta_x
        print 'u_lb:', self.ulb

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
        self.cs = None
        self.num_jumpers = None

        self.halo = None
        self.buf_nx = None
        self.buf_ny = None
        # For nonlocal computation
        self.psi_local_1 = None
        self.psi_local_2 = None

        self.allocate_constants()

        ## Initialize hydrodynamic variables & Shan-chen variables

        rho_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)

        u_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        v_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        self.u = cl.array.to_device(self.queue, u_host) # Velocity in the x direction; one per sim!
        self.v = cl.array.to_device(self.queue, v_host) # Velocity in the y direction; one per sim.

        u_prime_host = np.zeros((self.nx, self.ny), dtype=num_type, order='F')
        v_prime_host = np.zeros((self.nx, self.ny), dtype=num_type, order='F')
        self.u_prime = cl.array.to_device(self.queue, u_prime_host)  # Velocity in the x direction; one per sim!
        self.v_prime = cl.array.to_device(self.queue, v_prime_host)  # Velocity in the y direction; one per sim.

        # Intitialize the underlying feq equilibrium field
        feq_host = np.zeros((self.nx, self.ny, self.num_populations, self.num_jumpers), dtype=num_type, order='F')
        self.feq = cl.array.to_device(self.queue, feq_host)

        f_host = np.zeros((self.nx, self.ny, self.num_populations, self.num_jumpers), dtype=num_type, order='F')
        self.f = cl.array.to_device(self.queue, f_host)
        self.f_streamed = self.f.copy()

        # Initialize G: the body force acting on each phase
        Gx_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        Gy_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        self.Gx = cl.array.to_device(self.queue, Gx_host)
        self.Gy = cl.array.to_device(self.queue, Gy_host)

        #### COORDINATE SYSTEM: FOR CHECKING SIMULATIONS ####

        self.x_center = None
        self.y_center = None
        self.X_dim = None
        self.Y_dim = None

        self.x_center = self.nx / 2
        self.y_center = self.ny / 2

        xvalues = np.arange(self.nx)
        yvalues = np.arange(self.ny)
        Y, X = np.meshgrid(yvalues, xvalues)
        X = X.astype(num_type)
        Y = Y.astype(num_type)

        deltaX = X - self.x_center
        deltaY = Y - self.y_center

        # Convert to dimensionless coordinates
        self.X = deltaX / self.N
        self.Y = deltaY / self.N

        # Create list corresponding to all of the different fluids
        self.fluid_list = []
        self.tau_arr = []

        self.additional_collisions = [] # Takes into account growth, other things that can influence collisions
        self.additional_forces = []  # Takes into account other forces, i.e. surface tension

    def add_fluid(self, fluid):
        self.fluid_list.append(fluid)

    def complete_setup(self):
        # Run once all fluids have been added...gathers necessary info about the fluids

        # Generate the list of all relaxation times. Necessary to calculate
        # u and v prime.
        tau_host = []
        for cur_fluid in self.fluid_list:
            tau_host.append(cur_fluid.tau)
        tau_host = np.array(tau_host, dtype=num_type)
        print 'tau array:', tau_host
        self.tau_arr = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=tau_host)

        # Now calculate u and v prime
        self.update_velocity_prime()

    def update_velocity_prime(self):
        self.kernels.update_velocity_prime(
            self.queue, self.two_d_global_size, self.two_d_local_size,
            self.u_prime.data, self.v_prime.data,
            self.rho.data,
            self.f.data,
            self.tau_arr,
            self.w, self.cx, self.cy,
            self.nx, self.ny,
            self.num_populations, self.num_jumpers
        ).wait()


    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """
        self.nx = int_type(np.round(self.N*self.Lx))
        self.ny = int_type(np.round(self.N*self.Ly))

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
        self.kernels = cl.Program(self.context, open(file_dir + '/single_component.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """

        ##########################
        ##### D2Q9 parameters ####
        ##########################
        w = np.array([4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36.,
                      1. / 36., 1. / 36., 1. / 36.], order='F', dtype=num_type)  # weights for directions
        cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=int_type)  # direction vector for the x direction
        cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=int_type)  # direction vector for the y direction
        self.cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice

        self.num_jumpers = int_type(9)  # Number of jumpers for the D2Q9 lattice: 9

        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        # Allocate local memory for the clumpiness
        self.halo = int_type(1) # As we are doing D2Q9, we have a halo of one
        self.buf_nx = int_type(self.two_d_local_size[0] + 2 * self.halo)
        self.buf_ny = int_type(self.two_d_local_size[1] + 2 * self.halo)

        self.psi_local_1 = cl.LocalMemory(num_size * self.buf_nx * self.buf_ny)
        self.psi_local_2 = cl.LocalMemory(num_size * self.buf_nx * self.buf_ny)

    def add_eating_rate(self, eater_index, eatee_index, rate):
        """
        Eater eats eatee at a given rate.
        :param eater:
        :param eatee:
        :param rate:
        :return:
        """

        kernel_to_run = self.kernels.add_eating_collision
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(eater_index), int_type(eatee_index), num_type(rate),
            self.f.data, self.rho.data,
            self.w, self.cx, self.cy,
            self.nx, self.ny, self.num_populations, self.num_jumpers,
            self.delta_t, self.cs\
        ]

        self.additional_collisions.append([kernel_to_run, arguments])

    def add_constant_body_force(self, fluid_index, force_x, force_y):

        kernel_to_run = self.kernels.add_constant_body_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), num_type(force_x), num_type(force_y),
            self.Gx.data, self.Gy.data,
            self.nx, self.ny,
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_radial_body_force(self, fluid_index, center_x, center_y, prefactor, radial_scaling):

        kernel_to_run = self.kernels.add_radial_body_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), int_type(center_x), int_type(center_y),
            num_type(prefactor), num_type(radial_scaling),
            self.Gx.data, self.Gy.data,
            self.nx, self.ny,
            self.delta_x
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_interaction_force(self, fluid_1_index, fluid_2_index, G_int, bc='periodic'):

        kernel_to_run = self.kernels.add_interaction_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_1_index), int_type(fluid_2_index), num_type(G_int),
            self.psi_local_1, self.psi_local_2,
            self.rho.data, self.Gx.data, self.Gy.data,
            self.cs, self.cx, self.cy, self.w,
            self.nx, self.ny,
            self.buf_nx, self.buf_ny, self.halo, self.num_jumpers,
            self.delta_x
        ]

        if bc is 'periodic':
            arguments += [int_type(1), int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(0), int_type(1)]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_interaction_force_second_belt(self, fluid_1_index, fluid_2_index, G_int, bc='periodic'):

        #### pi1 ####
        pi1 = []
        cx1 = []
        cy1 = []

        c_temp = [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ]

        for c_vec in c_temp:
            pi1.append(4./63.)
            cx1.append(c_vec[0])
            cy1.append(c_vec[1])

        c_temp = [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]

        for c_vec in c_temp:
            pi1.append(4./135.)
            cx1.append(c_vec[0])
            cy1.append(c_vec[1])

        num_jumpers_1 = int_type(len(pi1))

        #### pi2 ####
        pi2 = []
        cx2 = []
        cy2 = []

        c_temp = [
            [2, 0],
            [0, 2],
            [-2, 0],
            [0, -2]
        ]

        for c_vec in c_temp:
            pi2.append(1./180.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        c_temp = [
            [2, -1],
            [2, 1],
            [1, 2],
            [-1, 2],
            [-2, 1],
            [-2, -1],
            [-1, -2],
            [1, -2]
        ]

        for c_vec in c_temp:
            pi2.append(2./945.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        c_temp = [
            [2, 2],
            [-2, 2],
            [-2, -2],
            [2, -2]
        ]
        for c_vec in c_temp:
            pi2.append(1./15120.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        num_jumpers_2 = int_type(len(pi2))

        ### Finish setup ###

        pi1 = np.array(pi1, dtype=num_type)
        cx1 = np.array(cx1, dtype=int_type)
        cy1 = np.array(cy1, dtype=int_type)

        print pi1
        print cx1
        print cy1

        print num_jumpers_1

        pi2 = np.array(pi2, dtype=num_type)
        cx2 = np.array(cx2, dtype=int_type)
        cy2 = np.array(cy2, dtype=int_type)

        print

        print pi2
        print cx2
        print cy2

        print num_jumpers_2

        pi1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi1)
        cx1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx1)
        cy1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy1)

        pi2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi2)
        cx2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx2)
        cy2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy2)

        # Allocate local memory for the clumpiness
        cur_halo = int_type(2) # As we are doing D2Q9, we have a halo of one
        cur_buf_nx = int_type(self.two_d_local_size[0] + 2 * cur_halo)
        cur_buf_ny = int_type(self.two_d_local_size[1] + 2 * cur_halo)

        local_1 = cl.LocalMemory(num_size * cur_buf_nx * cur_buf_ny)
        local_2 = cl.LocalMemory(num_size * cur_buf_nx * cur_buf_ny)

        kernel_to_run = self.kernels.add_interaction_force_second_belt
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_1_index), int_type(fluid_2_index), num_type(G_int),
            local_1, local_2,
            self.rho.data, self.Gx.data, self.Gy.data,
            self.cs,
            pi1_const, cx1_const, cy1_const, num_jumpers_1,
            pi2_const, cx2_const, cy2_const, num_jumpers_2,
            self.nx, self.ny,
            cur_buf_nx, cur_buf_ny, cur_halo,
            self.delta_x
        ]

        if bc is 'periodic':
            arguments += [int_type(1), int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(0), int_type(1)]

        self.additional_forces.append([kernel_to_run, arguments])

    def run(self, num_iterations, debug=False):
        """
        Run the simulation for num_iterations. Be aware that the same number of iterations does not correspond
        to the same non-dimensional time passing, as delta_t, the time discretization, will change depending on
        your resolution.

        :param num_iterations: The number of iterations to run
        """
        for cur_iteration in range(num_iterations):
            if debug:
                print 'At beginning of iteration:'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.move() # Move all jumpers
            if debug:
                print 'After move'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.move_bcs() # Must move before applying BC
            if debug:
                print 'After move bcs'
                self.check_fields()

            self.update_velocity_prime()
            if debug:
                print 'After updating velocity-prime'
                self.check_fields()

            # Update forces here as appropriate
            for cur_fluid in self.fluid_list:
                cur_fluid.update_hydro() # Update the hydrodynamic variables
            if debug:
                print 'After updating hydro'
                self.check_fields()

            # Reset the total body force and add to it as appropriate
            self.Gx[...] = 0
            self.Gy[...] = 0
            for d in self.additional_forces:
                kernel = d[0]
                arguments = d[1]
                kernel(*arguments).wait()

            if debug:
                print 'After updating supplementary forces'
                self.check_fields()

            # Update other forces
            for cur_fluid in self.fluid_list:
                cur_fluid.update_forces()  # Update the forces; some are based on the hydro
            if debug:
                print 'After updating internal forces'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.update_feq() # Update the equilibrium fields
            if debug:
                print 'After updating feq'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.collide_particles() # Relax the nonequilibrium fields.
            if debug:
                print 'After colliding particles'
                self.check_fields()

            # Loop over any additional collisions that are required (i.e. mass gain/loss)
            for d in self.additional_collisions:
                kernel = d[0]
                arguments = d[1]
                kernel(*arguments).wait()

    def check_fields(self):
        # Start with rho
        for i in range(self.num_populations):
            print 'Field:', i
            print 'rho_sum', cl.array.sum(self.rho[:, :, i])
            print 'f_sum', np.sum(self.f.get()[:, :, i, :])
            print 'f_eq_sum', np.sum(self.feq.get()[:, :, i, :])

        print 'Total rho_sum', cl.array.sum(self.rho)
        print 'Total f_sum', np.sum(self.f.get())
        print 'Total feq_sum', np.sum(self.feq.get())

        print
