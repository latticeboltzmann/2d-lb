import pyopencl as cl
import gpyfft as gfft

class Screened_Poisson(object):
    def __init__(self, cl_context, cl_queue, charge, lam=1., dx=1.):
        self.charge = charge
        self.lam = lam
        self.dx = dx

