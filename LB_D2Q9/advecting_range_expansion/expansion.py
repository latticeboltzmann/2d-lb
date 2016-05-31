import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct

# Required to draw obstacles
import skimage as ski
import skimage.draw
import skimage.morphology

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

class Expansion(object):
    """
    Simulates pipe flow using the D2Q9 lattice. Generally used to verify that our simulations were working correctly.
    For usage, see the docs folder.
    """

    def __init__(self, Lx=1.0, Ly=1.0, D=1.0, z=0.1,
                 vx=0., vy=0., vc=0.,
                 mu_list=None, Nb=10., Dc=1.0,
                 time_prefactor=1., N=50, rho_amp=1.0, zero_cutoff_factor = 0.01,
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
        self.phys_z = z # The size of the droplet

        self.phys_vx = vx
        self.phys_vy = vy
        self.phys_vc = vc

        self.phys_Nb = Nb
        self.phys_Dc = Dc

        self.phys_mu_list = np.array(mu_list)
        self.num_populations = np.int32(len(self.phys_mu_list))

        self.rho_amp = rho_amp
        self.zero_cutoff = np.float32(zero_cutoff_factor * (1./self.phys_Nb))

        # Interop with OpenGL?
        self.use_interop = use_interop

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

        self.dim_Pe = None # One number
        self.dim_G = None # An array
        self.dim_Dg = None # An array
        self.dim_D_population = None # An array as well
        self.dim_D_nutrient = None # A single number

        self.lb_G = None
        self.lb_Dg = None
        self.lb_D_population = None
        self.lb_D_nutrient = None

        self.omega = None
        self.omega_nutrient = None

        self.set_field_constants()

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
        self.random_generator = None
        self.random_normal = None

        self.lb_G_buf = None
        self.lb_Dg_buf = None
        self.lb_D_population_buf = None
        self.omega_buf = None

        self.allocate_constants()

        ## Initialize hydrodynamic variables
        # Assumes the same velocity field is shared by each component of the simulation
        self.u = None # The simulation's velocity in the x direction (horizontal)
        self.v = None # The simulation's velocity in the y direction (vertical)

        self.x_center = None
        self.y_center = None
        self.X_dim = None
        self.Y_dim = None

        # The last index is the nutrient field concentration
        self.rho = None
        self.init_hydro() # Create the hydrodynamic fields

        # Allocate the equilibrium distribution fields...the same for populations and nutrient fields
        feq_host = np.zeros((self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS), dtype=np.float32, order='F')
        self.feq = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=feq_host)
        # Create feq based on the initial hydrodynamic variables
        self.update_feq()

        # Now that feq is created, make the streaming fields
        self.f = None
        self.init_f()

        # Create a temporary buffer for streaming in parallel. Only need one.
        f_temp_host = np.zeros((self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS), dtype=np.float32, order='F')
        self.f_temporary = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=f_temp_host)

    def set_field_constants(self):
        # Note that diffusion is basically constant as a function of grid size, as delta_t ~ delta_x**2.

        self.dim_Pe = self.phys_z*self.phys_vc/self.phys_D
        print 'Pe:', self.dim_Pe

        self.dim_G = (self.phys_z**2/self.phys_D)*self.phys_mu_list
        print 'G:', self.dim_G
        self.lb_G = self.dim_G * self.delta_t
        self.lb_G = self.lb_G.astype(np.float32, order='F')
        print 'lb_G:', self.lb_G

        self.dim_Dg = (self.phys_mu_list/self.phys_Nb)*(1./self.phys_D)
        print 'Dg:', self.dim_Dg
        #self.lb_Dg = self.dim_Dg * (self.delta_t / self.delta_x ** 2)
        self.lb_Dg = self.dim_Dg * self.delta_t
        self.lb_Dg = self.lb_Dg.astype(np.float32, order='F')
        print 'lb_Dg:', self.lb_Dg
        print 'sqrt(lb_Dg):', np.sqrt(self.lb_Dg)

        self.dim_D_population = 1.0 * np.ones_like(self.phys_mu_list, dtype=np.float, order='F')
        print 'dim_D_populations:', self.dim_D_population
        self.lb_D_population = self.dim_D_population * (self.delta_t / self.delta_x ** 2)
        self.lb_D_population = self.lb_D_population.astype(np.float32, order='F')

        self.omega = (.5 + self.lb_D_population/cs**2)**-1.  # The relaxation time of the jumpers in the simulation
        self.omega = self.omega.astype(np.float32, order='F')
        print 'omega populations:', self.omega

        self.dim_D_nutrient = self.phys_Dc/self.phys_D
        print 'dim_D_concentration:', self.dim_D_nutrient

        self.lb_D_nutrient = self.dim_D_nutrient * (self.delta_t / self.delta_x ** 2)
        self.omega_nutrient = np.float32((.5 + self.lb_D_nutrient/cs**2)**-1.)
        print 'omega nutrient:', self.omega_nutrient

        print 'delta_ts:', self.dim_Dg/self.dim_G**2
        print 'our delta_t:', self.delta_t

    def set_characteristic_length_time(self):
        """
        Based on the input parameters, set the characteristic length and time scales. Required to make
        the simulation dimensionless. See http://www.latticeboltzmann.us/home/model-verification for more details.
        For pipe flow, L is the physical diameter of the pipe, and T is the time it takes the fluid moving at its
        theoretical maximum to to move a distance of L.
        """
        self.L = self.phys_z
        self.T = self.phys_z**2/self.phys_D

    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """

        self.lx = np.int32(self.N*int(self.phys_Lx/self.L))
        self.ly = np.int32(self.N*int(self.phys_Ly/self.L))

        self.nx = np.int32(self.lx + 2) # Total size of grid in x including boundaries
        self.ny = np.int32(self.ly + 2) # Total size of grid in y including boundaries

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
        self.kernels = cl.Program(self.context, open(parent_dir + '/D2Q9_multifield_diffusion.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """
        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        self.lb_G_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.lb_G)
        self.lb_Dg_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.lb_Dg)
        self.lb_D_population_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.lb_D_population)
        self.omega_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.omega)

        # Create a random generator
        self.random_generator = cl.clrandom.PhiloxGenerator(self.context)
        # Draw random normals for each population
        random_host = np.ones((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        self.random_normal = cl.array.to_device(self.queue, random_host)

        self.random_generator.fill_normal(self.random_normal, queue=self.queue)
        self.random_normal.finish()

    def init_hydro(self):
        """
        Based on the initial conditions, initialize the hydrodynamic fields, like density and velocity
        """

        nx = self.nx
        ny = self.ny

        #### COORDINATE SYSTEM: FOR CHECKING SIMULATIONS ####
        self.x_center = nx/2
        self.y_center = ny/2

        # Now initialize the gaussian
        xvalues = np.arange(nx)
        yvalues = np.arange(ny)
        X, Y = np.meshgrid(xvalues, yvalues)
        X = X.astype(np.float)
        Y = Y.astype(np.float)

        deltaX = X - self.x_center
        deltaY = Y - self.y_center

        # Convert to dimensionless coordinates
        self.X_dim = deltaX / self.N
        self.Y_dim = deltaY / self.N

        #### DENSITY #####

        # Last density is the nutrient field, always.
        rho = np.zeros((self.nx, self.ny, self.num_populations + 1), dtype=np.float32, order='F')

        # We must initialize each population density now. Just create an appropriate random array, and then take a
        # slice of it. For N populations, you need N-1 random fields.
        random_saturated_fields = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        random_saturated_fields[:, :, :] = np.random.rand(self.nx, self.ny, self.num_populations)
        # Normalize to one...it is saturated.
        sum_of_fields = random_saturated_fields.sum(axis = 2)

        random_saturated_fields /= sum_of_fields[:, :, np.newaxis]
        random_saturated_fields *= self.rho_amp

        circular_mask = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        rr, cc = ski.draw.circle(self.nx/2, self.ny/2, self.N, shape=circular_mask.shape)
        circular_mask[rr, cc] = 1.0

        random_saturated_fields *= circular_mask[:, :, np.newaxis]

        rho[:, :, 0:self.num_populations] = random_saturated_fields

        # Initialize nutrient field to one for now
        rho[:, :, self.num_populations] = 1.0
        rho = rho.astype(np.float32, order='F')
        self.rho = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=rho)

        #### VELOCITY ####
        dim_vx = self.dim_Pe * self.phys_vx / self.phys_vc
        dim_vy = self.dim_Pe * self.phys_vy / self.phys_vc

        lb_vx = (self.delta_t / self.delta_x) * dim_vx
        lb_vy = (self.delta_t / self.delta_x) * dim_vy

        u_host = lb_vx * np.ones((nx, ny), dtype=np.float32, order='F')  # Constant fluid motion for now
        v_host = lb_vy * np.ones((nx, ny), dtype=np.float32, order='F')

        # Transfer arrays to the device
        self.u = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=u_host)
        self.v = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_host)

    def update_feq(self):
        """
        Loop over and update feq for each field.
        """
        # feq is the same for populations and nutrients...updated together
        self.kernels.update_feq(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.feq, self.rho,
                                self.u, self.v,
                                self.w, self.cx, self.cy,
                                cs, self.nx, self.ny, self.num_populations,
                                self.zero_cutoff).wait()

    def init_f(self, amplitude = 0.001):
        """Requires init_feq to be run first."""
        f = np.zeros((self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.feq, is_blocking=True)

        # We now slightly perturb f
        perturb = (1. + amplitude * np.random.randn(self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS))
        f *= perturb

        # Now send f to the GPU
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)


    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        Implemented in OpenCL.
        """
        pass # A placeholder, needs to be fixed in the future...

    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """

        # Annoyingly, f is now four-dimensional. (nx, ny, num_populations +1, NUM_JUMPERS)
        self.kernels.move(self.queue, self.two_d_global_size, self.two_d_local_size,
                          self.f, self.f_temporary,
                          self.cx, self.cy,
                          self.nx, self.ny, self.num_populations).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        self.kernels.copy_buffer(self.queue, self.two_d_global_size, self.two_d_local_size,
                                 self.f_temporary, self.f,
                                 self.nx, self.ny, self.num_populations).wait()

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        """
        self.kernels.update_hydro(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f, self.u, self.v, self.rho,
                                self.nx, self.ny, self.num_populations).wait()

    def collide_particles(self):
        """
        Relax the nonequilibrium f fields towards their equilibrium feq. Depends on omega. Implemented in OpenCL.
        """

        # We need to deal with the subpopulations and the concentration field separately.

        self.kernels.collide_particles(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f, self.feq, self.rho, self.random_normal.data,
                                self.omega_buf, self.lb_G_buf, self.lb_Dg_buf,
                                self.omega_nutrient,
                                self.w, self.nx, self.ny, self.num_populations,
                                self.zero_cutoff).wait()

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

            self.update_hydro() # Update the hydrodynamic variables based on values of f
            self.update_feq() # Update the equilibrium fields
            self.collide_particles() # Relax the nonequilibrium fields. Reaction takes place here.

            # Regenerate random fields in each subpopulation
            self.random_generator.fill_normal(self.random_normal, queue=self.queue)
            self.random_normal.finish()

    def get_fields(self):
        """
        :return: Returns a dictionary of all fields. Transfers data from the GPU to the CPU.
        """
        f = np.zeros((self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.f, is_blocking=True)

        feq = np.zeros((self.nx, self.ny, self.num_populations + 1, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, feq, self.feq, is_blocking=True)

        u = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, u, self.u, is_blocking=True)

        v = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, v, self.v, is_blocking=True)

        rho = np.zeros((self.nx, self.ny, self.num_populations + 1), dtype=np.float32, order='F')
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