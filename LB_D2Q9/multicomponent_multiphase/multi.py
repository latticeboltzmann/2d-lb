import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
import matplotlib.pyplot as plt
from LB_D2Q9.spectral_poisson import screened_poisson as sp

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

class Fluid(object):

    def __init__(self, sim, field_index, nu = 1.0, bc='periodic'):

        self.sim = sim # TODO: MAKE THIS A WEAKREF

        self.field_index = int_type(field_index)

        self.lb_nu_e = num_type(nu)
        self.bc = bc

        # Determine the viscosity
        self.tau = num_type(.5 + self.lb_nu_e / (sim.cs**2))
        print 'tau', self.tau
        self.omega = num_type(self.tau ** -1.)  # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.


    def initialize(self, rho_arr, f_amp = 0.0):
        """
        ASSUMES THAT THE BARYCENTRIC VELOCITY IS ALREADY SET
        """

        #### DENSITY #####
        rho_host = self.sim.rho.get()

        rho_host[:, :, self.field_index] = rho_arr
        self.sim.rho = cl.array.to_device(self.sim.queue, rho_host)

        #### UPDATE HOPPERS ####
        self.update_feq() # Based on the hydrodynamic fields, create feq

        # Now initialize the nonequilibrium f
        self.init_pop(amplitude=f_amp) # Based on feq, create the hopping non-equilibrium fields

        #### Update the component velocities & resulting forces ####
        # self.update_hydro() # The point here is to initialize u and v, the component velocities
        # self.update_forces() # Calculates the drag force, if necessary, based on the component velocity

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
        """For internal forces...none in this case."""

        pass

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_feq_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.feq.data,
            sim.rho.data,
            sim.u_bary.data, sim.v_bary.data,
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

        sim = self.sim

        sim.kernels.update_hydro_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            sim.Gx.data, sim.Gy.data,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers
        ).wait()

        if sim.check_max_ulb:
            max_ulb = cl.array.max((sim.u[:, :, self.field_index]**2 + sim.v[:, :, self.field_index]**2)**.5, queue=sim.queue)

            if max_ulb > sim.cs*sim.mach_tolerance:
                print 'max_ulb is greater than cs/10! Ma=', max_ulb/sim.cs

    def collide_particles(self):
        sim = self.sim

        self.sim.kernels.collide_particles_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.feq.data,
            sim.rho.data,
            sim.u_bary.data, sim.v_bary.data,
            sim.Gx.data, sim.Gy.data,
            self.omega,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers,
            sim.cs
        ).wait()

class Simulation_Runner(object):
    """
    Everything is in dimensionless units. It's just easier.
    """

    def __init__(self, nx=100, ny=100,
                 L_lb=100, T_lb=1.,
                 num_populations=1,
                 two_d_local_size=(32,32), use_interop=False,
                 check_max_ulb=False, mach_tolerance=0.1,
                 context = None):

        self.nx = int_type(nx)
        self.ny = int_type(ny)

        self.L_lb = int_type(L_lb) # The resolution of the simulation
        self.T_lb = num_type(T_lb) # How many steps it takes to reach T=1

        self.delta_x = 1./self.L_lb
        self.delta_t = 1./self.T_lb

        # Book-keeping
        self.num_populations = int_type(num_populations)

        self.check_max_ulb = check_max_ulb
        self.mach_tolerance = mach_tolerance

        # Create global & local sizes appropriately
        self.two_d_local_size = two_d_local_size        # The local size to be used for 2-d workgroups
        self.two_d_global_size = get_divisible_global((self.nx, self.ny), self.two_d_local_size)

        print '2d global:' , self.two_d_global_size
        print '2d local:' , self.two_d_local_size

        # Initialize the opencl environment
        self.context = context     # The pyOpenCL context
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

        self.allocate_constants()

        ## Initialize hydrodynamic variables & Shan-chen variables

        rho_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)

        u_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        v_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=num_type, order='F')
        self.u = cl.array.to_device(self.queue, u_host) # Velocity in the x direction; one per sim!
        self.v = cl.array.to_device(self.queue, v_host) # Velocity in the y direction; one per sim.

        u_bary_host = np.zeros((self.nx, self.ny), dtype=num_type, order='F')
        v_bary_host = np.zeros((self.nx, self.ny), dtype=num_type, order='F')
        self.u_bary = cl.array.to_device(self.queue, u_bary_host)  # Velocity in the x direction; one per sim!
        self.v_bary = cl.array.to_device(self.queue, v_bary_host)  # Velocity in the y direction; one per sim.

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

        # Create list corresponding to all of the different fluids
        self.fluid_list = []
        self.tau_arr = []

        self.additional_collisions = [] # Takes into account growth, other things that can influence collisions
        self.additional_forces = []  # Takes into account other forces, i.e. surface tension

        self.poisson_solver = None # To solve the poisson & screened poisson equation, if necessary.
        self.poisson_force_active = False
        self.poisson_source_index = None
        self.poisson_force_index = None
        self.poisson_amp = None
        self.poisson_xgrad = None
        self.poisson_ygrad = None

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

    def set_bary_velocity(self, u_bary_host, v_bary_host):
        self.u_bary = cl.array.to_device(self.queue, u_bary_host)
        self.v_bary = cl.array.to_device(self.queue, v_bary_host)

    def update_bary_velocity(self):
        self.kernels.update_bary_velocity(
            self.queue, self.two_d_global_size, self.two_d_local_size,
            self.u_bary.data, self.v_bary.data,
            self.rho.data,
            self.f.data,
            self.Gx.data, self.Gy.data,
            self.tau_arr,
            self.w, self.cx, self.cy,
            self.nx, self.ny,
            self.num_populations, self.num_jumpers
        ).wait()


    def init_opencl(self):
        """
        Initializes the base items needed to run OpenCL code.
        """

        # Startup script shamelessly taken from CS205 homework

        if self.context is None:
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
        self.kernels = cl.Program(self.context, open(file_dir + '/multi.cl').read()).build(options='')

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

    def add_eating_rate(self, eater_index, eatee_index, rate, orderparameter_cutoff):
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
            num_type(orderparameter_cutoff),
            self.f.data, self.rho.data,
            self.w, self.cx, self.cy,
            self.nx, self.ny, self.num_populations, self.num_jumpers,
            self.cs
        ]

        self.additional_collisions.append([kernel_to_run, arguments])


    def add_growth(self, eater_index, min_rho_cutoff, max_rho_cutoff, eat_rate):
        """
        Grows uniformly everywhere.
        """

        kernel_to_run = self.kernels.add_growth
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(eater_index),
            num_type(min_rho_cutoff), num_type(max_rho_cutoff),
            num_type(eat_rate),
            self.f.data, self.rho.data,
            self.w, self.cx, self.cy,
            self.nx, self.ny, self.num_populations, self.num_jumpers,
            self.cs
        ]

        self.additional_collisions.append([kernel_to_run, arguments])


    def add_constant_g_force(self, fluid_index, force_x, force_y):

        kernel_to_run = self.kernels.add_constant_g_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), num_type(force_x), num_type(force_y),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_radial_g_force(self, fluid_index, center_x, center_y, prefactor, radial_scaling):

        kernel_to_run = self.kernels.add_radial_g_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), int_type(center_x), int_type(center_y),
            num_type(prefactor), num_type(radial_scaling),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    ##### Dealing with Poisson Repulsion. ###########
    def add_screened_poisson_force(self, source_index, force_index, interaction_length, amplitude):

        input_density = self.rho.get()[:, :, source_index]
        self.poisson_solver = sp.Screened_Poisson(input_density, cl_context=self.context, cl_queue = self.queue,
                                                  lam=interaction_length, dx=1.0)
        self.poisson_solver.create_grad_fields()

        self.poisson_force_active = True
        self.poisson_source_index = int_type(source_index)
        self.poisson_force_index = int_type(force_index)
        self.poisson_amp = amplitude

    def screened_poisson_kernel(self):
        # Update the charge field for the poisson solver
        density_view = self.rho[:, :, self.poisson_source_index]

        cl.enqueue_copy(self.queue, self.poisson_solver.charge.data, density_view.astype(np.complex64).data)

        self.poisson_solver.solve_and_update_grad_fields()
        self.poisson_xgrad = self.poisson_amp * self.poisson_solver.xgrad.real
        self.poisson_ygrad = self.poisson_amp * self.poisson_solver.ygrad.real

        self.Gx[:, :, self.poisson_force_index] += self.poisson_xgrad
        self.Gy[:, :, self.poisson_force_index] += self.poisson_ygrad
    ############################################

    def add_interaction_force(self, fluid_1_index, fluid_2_index, G_int, bc='periodic', potential='linear',
                              potential_parameters=None):

        # We use the D2Q9 stencil for this force
        w_arr = np.array([4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36.,
                      1. / 36., 1. / 36., 1. / 36.], order='F', dtype=num_type)  # weights for directions
        cx_arr = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=int_type)  # direction vector for the x direction
        cy_arr = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=int_type)  # direction vector for the y direction

        w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_arr)
        cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx_arr)
        cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy_arr)

        cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice
        num_jumpers = int_type(9)  # Number of jumpers for the D2Q9 lattice: 9

        # Allocate local memory
        halo = int_type(1) # As we are doing D2Q9, we have a halo of one
        buf_nx = int_type(self.two_d_local_size[0] + 2 * halo)
        buf_ny = int_type(self.two_d_local_size[1] + 2 * halo)


        psi_local_1 = cl.LocalMemory(num_size * buf_nx * buf_ny)
        psi_local_2 = cl.LocalMemory(num_size * buf_nx * buf_ny)

        kernel_to_run = self.kernels.add_interaction_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_1_index), int_type(fluid_2_index), num_type(G_int),
            psi_local_1, psi_local_2,
            self.rho.data, self.Gx.data, self.Gy.data,
            cs, cx, cy, w,
            self.nx, self.ny,
            buf_nx, buf_ny, halo, num_jumpers
        ]

        if bc is 'periodic':
            arguments += [int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(1)]
        else:
            raise ValueError('Specified boundary condition does not exist')

        if potential is 'linear':
            arguments += [int_type(0)]
        elif potential is 'shan_chen':
            arguments += [int_type(1)]
        elif potential is 'pow':
            arguments += [int_type(2)]
        elif potential is 'vdw':
            arguments += [int_type(3)]
        else:
            raise ValueError('Specified pseudopotential does not exist.')

        if potential_parameters is None:
            potential_parameters = np.array([0.], dtype=num_type)
        else:
            potential_parameters = np.array(potential_parameters, dtype=num_type)

        parameters_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=potential_parameters)

        arguments += [parameters_const]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_interaction_force_second_belt(self, fluid_1_index, fluid_2_index, G_int, bc='periodic', potential='linear',
                                          potential_parameters=None):

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

        pi2 = np.array(pi2, dtype=num_type)
        cx2 = np.array(cx2, dtype=int_type)
        cy2 = np.array(cy2, dtype=int_type)

        pi1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi1)
        cx1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx1)
        cy1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy1)

        pi2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi2)
        cx2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx2)
        cy2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy2)

        # Allocate local memory for the clumpiness
        cur_halo = int_type(2)
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
            cur_buf_nx, cur_buf_ny, cur_halo
        ]

        if bc is 'periodic':
            arguments += [int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(1)]
        else:
            raise ValueError('Specified boundary condition does not exist')

        if potential is 'linear':
            arguments += [int_type(0)]
        elif potential is 'shan_chen':
            arguments += [int_type(1)]
        elif potential is 'pow':
            arguments += [int_type(2)]
        elif potential is 'vdw':
            arguments += [int_type(3)]
        else:
            raise ValueError('Specified pseudopotential does not exist.')

        if potential_parameters is None:
            potential_parameters = np.array([0.], dtype=num_type)
        else:
            potential_parameters = np.array(potential_parameters, dtype=num_type)

        parameters_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=potential_parameters)

        arguments += [parameters_const]

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
            if self.poisson_force_active:
                self.screened_poisson_kernel()
            if debug:
                print 'After updating supplementary forces'
                self.check_fields()

            # Update other forces...includes pourous effects & must be run last
            for cur_fluid in self.fluid_list:
                cur_fluid.update_forces()
            if debug:
                print 'After updating internal forces'
                self.check_fields()

            # After updating forces, update the bary_velocity
            self.update_bary_velocity()
            if debug:
                print 'After updating bary-velocity'
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
            print 'u, v bary sum', cl.array.sum(self.u_bary), cl.array.sum(self.u_bary)
            print 'f_sum', np.sum(self.f.get()[:, :, i, :])
            print 'f_eq_sum', np.sum(self.feq.get()[:, :, i, :])

        print 'Total rho_sum', cl.array.sum(self.rho)
        print 'Total f_sum', np.sum(self.f.get())
        print 'Total feq_sum', np.sum(self.feq.get())

        print


class Simulation_RunnerD2Q25(Simulation_Runner):
    def __init__(self, **kwargs):
        super(Simulation_RunnerD2Q25, self).__init__(**kwargs)

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """

        ##########################
        ##### D2Q25 parameters####
        ##########################
        t0 = (4./45.)*(4 + np.sqrt(10))
        t1 = (3./80.)*(8 - np.sqrt(10))
        t3 = (1./720.)*(16 - 5*np.sqrt(10))

        w_list = []
        cx_list = []
        cy_list = []

        # Mag 0
        cx_list += [0]
        cy_list += [0]
        w_list += [t0*t0]

        # Mag 1
        cx_list += [0, 0, 1, -1]
        cy_list += [1, -1, 0, 0]
        w_list += 4*[t0*t1]

        # Mag sqrt(2)
        cx_list += [1, 1, -1, -1]
        cy_list += [1, -1, 1, -1]
        w_list += 4*[t1*t1]

        # Mag 3
        cx_list += [3, -3, 0, 0]
        cy_list += [0, 0, 3, -3]
        w_list += 4*[t0*t3]

        # Mag sqrt(10)
        cx_list += [1, 1, -1, -1, 3, 3, -3, -3]
        cy_list += [3, -3, 3, -3, 1, -1, 1, -1]
        w_list += 8*[t1*t3]

        # Mag sqrt(18)
        cx_list += [3, 3, -3, -3]
        cy_list += [3, -3, 3, -3]
        w_list += 4*[t3 * t3]

        # Now send everything to disk
        w = np.array(w_list, order='F', dtype=num_type)  # weights for directions
        cx = np.array(cx_list, order='F', dtype=int_type)  # direction vector for the x direction
        cy = np.array(cy_list, order='F', dtype=int_type)  # direction vector for the y direction

        self.cs = num_type(np.sqrt(1. - np.sqrt(2./5.)))  # Speed of sound on the lattice
        self.num_jumpers = int_type(w.shape[0])  # Number of jumpers: should be 25

        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)