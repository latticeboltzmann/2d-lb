import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import ctypes as ct

# Required to draw obstacles
import skimage as ski
import skimage.draw

float_size = ct.sizeof(ct.c_float)

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)

##########################
##### D2Q9 parameters ####
##########################
w=np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,
            1./36.,1./36.,1./36.], order='F', dtype=np.float32) # weights for directions
cx=np.array([0,1,0,-1,0,1,-1,-1,1], order='F', dtype=np.int32) # direction vector for the x direction
cy=np.array([0,0,1,0,-1,1,1,-1,-1], order='F', dtype=np.int32) # direction vector for the y direction
cs=1./np.sqrt(3)
cs2 = cs**2
cs22 = 2*cs2
two_cs4 = 2*cs**4

w0 = 4./9.
w1 = 1./9.
w2 = 1./36.

NUM_JUMPERS = 9

def get_divisible_global(global_size, local_size):
    new_size = []
    for cur_global, cur_local in zip(global_size, local_size):
        remainder = cur_global % cur_local
        if remainder == 0:
            new_size.append(cur_global)
        else:
            new_size.append(cur_global + cur_local - remainder)
    return tuple(new_size)

class Pipe_Flow(object):
    """2d pipe flow with D2Q9"""

    def set_characteristic_length_time(self):
        """Necessary for subclassing"""
        self.L = self.phys_diameter
        self.T = (8*self.phys_rho*self.phys_visc)/(np.abs(self.phys_pressure_grad)*self.L)

    def initialize_grid_dims(self):
        """Necessary for subclassing"""

        self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
        self.ly = self.N

        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

    def __init__(self, diameter=None, rho=None, viscosity=None, pressure_grad=None, pipe_length=None,
                 N=200, time_prefactor = 1.,
                 two_d_local_size=(32,32), three_d_local_size=(32,32,1)):

        # Physical units
        self.phys_diameter = diameter
        self.phys_rho = rho
        self.phys_visc = viscosity
        self.phys_pressure_grad = pressure_grad
        self.phys_pipe_length = pipe_length

        # Get the characteristic length and time scales for the flow
        self.L = None
        self.T = None
        self.set_characteristic_length_time()
        print 'Characteristic L:', self.L
        print 'Characteristic T:', self.T

        # Initialize the reynolds number
        self.Re = self.L**2/(self.phys_visc*self.T**2)
        print 'Reynolds number:', self.Re

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = 1./N # How many squares characteristic length is broken into
        self.delta_t = time_prefactor * self.delta_x**2 # How many time iterations until the characteristic time, should be ~ \delta x^2

        # Initialize grid dimensions
        self.lx = None
        self.ly = None
        self.nx = None
        self.ny = None
        self.initialize_grid_dims()

        # Get the non-dimensional pressure gradient
        nondim_deltaP = (self.T**2/(self.phys_rho*self.L))*self.phys_pressure_grad
        # Obtain the difference in density (pressure) at the inlet & outlet
        delta_rho = self.nx*(self.delta_t**2/self.delta_x)*(1./cs2)*nondim_deltaP

        # Assume deltaP is negative. So, outlet will have a smaller density.

        self.outlet_rho = 1.
        self.inlet_rho = 1. + np.abs(delta_rho)

        print 'inlet rho:' , self.inlet_rho
        print 'outlet rho:', self.outlet_rho

        self.lb_viscosity = (self.delta_t/self.delta_x**2) * (1./self.Re)

        # Get omega from lb_viscosity
        self.omega = (self.lb_viscosity/cs2 + 0.5)**-1.
        print 'omega', self.omega
        assert self.omega < 2.

        # Create global & local sizes appropriately
        self.two_d_local_size = two_d_local_size
        self.three_d_local_size = three_d_local_size

        self.two_d_global_size = get_divisible_global((self.nx, self.ny), self.two_d_local_size)
        self.three_d_global_size = get_divisible_global((self.nx, self.ny, 9), self.three_d_local_size)

        print '2d global:' , self.two_d_global_size
        print '2d local:' , self.two_d_local_size
        print '3d global:' , self.three_d_global_size
        print '3d local:' , self.three_d_local_size

        # Initialize the opencl environment
        self.context = None
        self.queue = None
        self.kernels = None
        self.init_opencl()

        # Allocate constants & local memory for opencl
        self.w = None
        self.cx = None
        self.cy = None
        self.local_u = None
        self.local_v = None
        self.local_rho = None
        self.allocate_constants()

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

    def allocate_constants(self):
        """Allocates constants to be used by opencl."""

        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        self.local_u = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])
        self.local_v = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])
        self.local_rho = cl.LocalMemory(float_size * self.two_d_local_size[0]*self.two_d_local_size[1])

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

    def init_hydro(self):
        nx = self.nx
        ny = self.ny

        # Initialize arrays on the host
        rho_host = self.inlet_rho*np.ones((nx, ny), dtype=np.float32, order='F')
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
        self.kernels.move_bcs(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f, self.u,
                                np.float32(self.inlet_rho), np.float32(self.outlet_rho),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def move(self):
        # Always copy f, then f_streamed
        self.kernels.move(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f, self.f_streamed,
                                self.cx, self.cy,
                                np.int32(self.nx), np.int32(self.ny)).wait()

        # Set f equal to f streamed. This way, if things do not stream, it is ok in future iterations.
        self.kernels.copy_buffer(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.f_streamed, self.f,
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def init_pop(self):
        nx = self.nx
        ny = self.ny

        # For simplicity, copy feq to the local host, where you can make a copy
        f = np.zeros((nx, ny, NUM_JUMPERS), dtype=np.float32, order='F')
        cl.enqueue_copy(self.queue, f, self.feq, is_blocking=True)

        # We now slightly perturb f
        amplitude = .001
        perturb = (1. + amplitude*np.random.randn(nx, ny, NUM_JUMPERS))
        f *= perturb

        # Now send f to the GPU
        self.f = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

        # Create a new buffer
        self.f_streamed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f)

    def update_hydro(self):
        self.kernels.update_hydro(self.queue, self.two_d_global_size, self.two_d_local_size,
                                self.f, self.u, self.v, self.rho,
                                np.float32(self.inlet_rho), np.float32(self.outlet_rho),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def update_feq(self):
        self.kernels.update_feq(self.queue, self.three_d_global_size, self.three_d_local_size,
                                self.feq,
                                self.u, self.v, self.rho,
                                self.local_u, self.local_v, self.local_rho,
                                self.w, self.cx, self.cy,
                                np.float32(cs), np.float32(cs2), np.float32(cs22), np.float32(two_cs4),
                                np.int32(self.nx), np.int32(self.ny)).wait()

    def collide_particles(self):
        self.kernels.collide_particles(self.queue, self.three_d_global_size, self.three_d_local_size,
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

    def get_fields(self):
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
        fields = self.get_fields()

        fields['u'] *= self.delta_x/self.delta_t
        fields['v'] *= self.delta_x/self.delta_t

        return fields

    def get_physical_fields(self):
        fields = self.get_nondim_fields()

        fields['u'] *= (self.L/self.T)
        fields['v'] *= (self.L/self.T)

        return fields


class Pipe_Flow_Cylinder(Pipe_Flow):

    def set_characteristic_length_time(self):
        """Necessary for subclassing"""
        self.L = self.phys_cylinder_radius
        self.T = (8*self.phys_rho*self.phys_visc*self.L)/(np.abs(self.phys_pressure_grad)*self.phys_diameter**2)

    def initialize_grid_dims(self):
        """Necessary for subclassing"""

        self.lx = int(np.ceil((self.phys_pipe_length / self.L)*self.N))
        self.ly = int(np.ceil((self.phys_diameter / self.L)*self.N))

        self.nx = self.lx + 1 # Total size of grid in x including boundary
        self.ny = self.ly + 1 # Total size of grid in y including boundary

        ## Initialize the obstacle mask
        self.obstacle_mask_host = np.zeros((self.nx, self.ny), dtype=np.int32, order='F')

        # Initialize the obstacle in the correct place
        x_cylinder = self.N * self.phys_cylinder_center[0]/self.L
        y_cylinder = self.N * self.phys_cylinder_center[1]/self.L

        circle = ski.draw.circle(x_cylinder, y_cylinder, self.N)
        self.obstacle_mask_host[circle[0], circle[1]] = 1


    def __init__(self, cylinder_center = None, cylinder_radius=None, **kwargs):
        """Obstacle mask should be ones and zeros."""

        assert (cylinder_center is not None)
        assert (cylinder_radius is not None) # If there are no obstacles, this will definitely not run.

        self.phys_cylinder_center = cylinder_center
        self.phys_cylinder_radius = cylinder_radius

        self.obstacle_mask_host = None
        self.obstacle_mask = None
        super(Pipe_Flow_Cylinder, self).__init__(**kwargs)


    def init_hydro(self):
        super(Pipe_Flow_Cylinder, self).init_hydro()

        # Now create the obstacle mask on the device
        self.obstacle_mask = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self.obstacle_mask_host)

        # Based on where the obstacle mask is, set velocity to zero, as appropriate.

        self.kernels.set_zero_velocity_in_obstacle(self.queue, self.two_d_global_size, self.two_d_local_size,
                                                   self.obstacle_mask, self.u, self.v,
                                                   np.int32(self.nx), np.int32(self.ny)).wait()

    def update_hydro(self):
        super(Pipe_Flow_Cylinder, self).update_hydro()
        self.kernels.set_zero_velocity_in_obstacle(self.queue, self.two_d_global_size, self.two_d_local_size,
                                                   self.obstacle_mask, self.u, self.v,
                                                   np.int32(self.nx), np.int32(self.ny)).wait()

    def move_bcs(self):
        super(Pipe_Flow_Cylinder, self).move_bcs()

        # Now bounceback on the obstacle
        self.kernels.bounceback_in_obstacle(self.queue, self.two_d_global_size, self.two_d_local_size,
                                            self.obstacle_mask, self.f,
                                            np.int32(self.nx), np.int32(self.ny)).wait()