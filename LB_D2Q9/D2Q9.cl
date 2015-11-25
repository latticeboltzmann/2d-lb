const float[9] w = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36., 1./36.,1./36.,1./36.]}; // weights for directions
const int[9]  cx= {0,1,0,-1,0,1,-1,-1,1]} // direction vector for the x direction
const int [9] cy = {0,0,1,0,-1,1,1,-1,-1} // direction vector for the y direction

const float cs = 1/np.sqrt(3);
const float cs2 = cs**2;
const float two_cs2 = 2.*cs2;
const float two_cs4 = 2*cs**4;

const float NUM_JUMPERS = 9;

__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __global __read_only float *rho_global,
           const int nx, const int ny)
{
    //Luckily, everything takes place inplace, so this isn't too bad. No local buffers needed.
    //First dimension should be x, second dimension y, third dimension jumper type
    //Note that this is different from how things are structured now

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    float u = u_global[x, y];
    float v= v_global[x, y];
    float rho = rho_global[x, y];

    // We used to have a bunch of if statements here. It's better to have something thata
    // can be executed in parallel.
    cdef float cur_w = w[jump_id];
    cdef int cur_cx = cx[jump_id];
    cdef int cur_cy = cy[jump_id];

    cdef float cur_c_dot_u = cur_cx*u + cur_cy*v;
    cdef float velocity_squared = u*u + v*v;

    cdef float inner_feq = 1 + cur_c_dot_u/cs2 + cur_c_dot_u**2/two_cs4 - velocity_squared/two_cs2;

    cdef float new_feq =  cur_w*rho*inner_feq

    feq_global[x, y, jump_id] = new_feq
}