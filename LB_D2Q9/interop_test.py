# Initialize a simulation without interop...we will redo this soon.

import numpy as np
from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app
import pyopencl as cl
from LB_D2Q9.dimensionless import opencl_dim as lb_cl

D = 10.5 # meter
rho = 10. # kg/m^3
nu = 5. # Viscosity, m^2/s

pressure_grad = -100 # Pa/m
pipe_length = 2*D # meter
N = 300
setup_sim = lb_cl.Pipe_Flow(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                            N=N, time_prefactor=1,
                            two_d_local_size=(32, 32), three_d_local_size=(32, 32, 1))

# Image to be displayed
W, H = setup_sim.nx, setup_sim.ny
I = np.zeros((W, H), dtype=np.float32, order='F')
cl.enqueue_copy(setup_sim.queue, I, setup_sim.u, is_blocking=True)

# A simple texture quad
data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                          ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

VERT_SHADER = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=((W * 5), (H * 5)))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(I, interpolation='linear')

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        gloo.set_clear_color('white')

        self._timer = app.Timer('auto', connect=self.update, start=True)

        self.sim_is_initialized = False
        self.sim = None

        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

        # Compute thje new size of the quad
        r = width / float(height)
        R = W / float(H)
        if r < R:
            w, h = width, width / R
            x, y = 0, int((height - h) / 2)
        else:
            w, h = height * R, height
            x, y = int((width - w) / 2), 0
        data['a_position'] = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(gloo.VertexBuffer(data))

    def initialize_sim(self):
        # Now that OpenGL is setup, we can initialize interop correctly...
        self.sim_is_initialized = True
        self.sim = lb_cl.Pipe_Flow(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                              N=N, time_prefactor=1,
                              two_d_local_size=(32, 32), three_d_local_size=(32, 32, 1), use_interop=True)
        print self.texture.id
        self.texture.glir
        gl_textureBuf = cl.GLTexture(self.sim.context, cl.mem_flags.READ_WRITE,
                                     gloo.gl.GL_TEXTURE_2D, 0, self.texture.id)
        # Replace "v" with the GL texture
        self.sim.v = gl_textureBuf

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        if not self.sim_is_initialized:
            self.texture.set_data(I)
            self.program.draw('triangle_strip')
            self.update()
            self.initialize_sim()
        else:
            cl.enqueue_acquire_gl_objects(self.sim.queue, [self.sim.v])
            self.sim.run(1)
            cl.enqueue_release_gl_objects(self.sim.queue, [self.sim.v])
            self.texture.set_data(I)
            self.program.draw('triangle_strip')

# Now actually run things
c = Canvas()
app.run()