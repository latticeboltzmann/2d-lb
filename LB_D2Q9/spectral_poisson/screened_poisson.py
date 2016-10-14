import pyopencl as cl
import pyopencl.array
import gpyfft as gfft
import numpy as np


class Screened_Poisson(object):
    def __init__(self, charge, cl_context=None, cl_queue=None, lam=1., dx=1.):
        self.context = cl_context
        self.queue = cl_queue

        if self.context is None:
            self.create_context_and_queue()

        charge_cpu = charge.astype(np.complex64)
        self.charge = cl.array.to_device(self.queue, charge_cpu)

        self.transform = gfft.fft.FFT(self.context, self.queue, (charge,), axes=(0, 1))

        self.lam = lam # Interaction length lambda
        self.dx = dx # Spatial scale

        Lx = self.dx * charge_cpu.shape[0]
        Ly = self.dx * charge_cpu.shape[1]

        freq_x = Lx * np.fft.fftfreq(charge_cpu.shape[0], d=dx)
        freq_y = Ly * np.fft.fftfreq(charge_cpu.shape[1], d=dx)

        freq_Y_cpu, freq_X_cpu = np.meshgrid(freq_y, freq_x)
        # Calculate the rescaling on the CPU, as it only has to be done once.

        self.freq_X = cl.array.to_device(freq_X_cpu)
        self.freq_Y = cl.array.to_device(freq_Y_cpu)

        rescaling_cpu = 1./(self.lam**2*(self.freq_X**2 + self.freq_Y**2) + 1.)

        self.rescaling = cl.array.to_device(self.queue, rescaling_cpu)

        self.xgrad = None
        self.ygrad = None

        self.xgrad_transform = None
        self.ygrad_transform = None


    def fft_and_screen(self):
        event, = self.transform.enqueue()
        event.wait()

        self.charge *= self.rescaling

    def inverse_fft(self):
        event, = self.transform.enqueue(forward=False)
        event.wait()

    def create_grad_fields(self):
        xgrad_cpu = np.zeros_like(self.charge)
        ygrad_cpu = np.zeros_like(self.charge)

        self.xgrad = cl.array.to_device(self.queue, xgrad_cpu)
        self.ygrad = cl.array.to_device(self.queue, ygrad_cpu)

    def create_context_and_queue(self):
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
        self.context = cl.Context(devices)
        print 'This context is associated with ', self.context.devices, 'devices'

        # Create a simple queue
        self.queue = cl.CommandQueue(self.context, self.context.devices[0],
                                properties=cl.command_queue_properties.PROFILING_ENABLE)