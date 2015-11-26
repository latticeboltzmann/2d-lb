
__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __global __read_only float *rho_global,
           int nx, int ny)
{
    //Define global constants...which openCL makes difficult!
    const float w[9] = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36., 1./36.,1./36.,1./36.}; // weights for directions
    const int cx[9] = {0,1,0,-1,0,1,-1,-1,1}; // direction vector for the x direction
    const int cy[9] = {0,0,1,0,-1,1,1,-1,-1}; // direction vector for the y direction

    const float cs = 1/pow(3.f, .5f);
    const float cs2 = pow(cs, 2.f);
    const float two_cs2 = 2.*cs2;
    const float two_cs4 = 2*pow(cs, 4.f);

    //Luckily, everything takes place inplace, so this isn't too bad. No local buffers needed.
    //First dimension should be x, second dimension y, third dimension jumper type
    //Note that this is different from how things are structured now

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);
    //The position in 3d is confusing...

    int index_position = jump_id*nx*ny + y*nx + x;

    float u = u_global[index_position];
    float v= v_global[index_position];
    float rho = rho_global[index_position];

    // We used to have a bunch of if statements here. It's better to have something thata
    // can be executed in parallel.
    float cur_w = w[jump_id];
    int cur_cx = cx[jump_id];
    int cur_cy = cy[jump_id];

    float cur_c_dot_u = cur_cx*u + cur_cy*v;
    float velocity_squared = u*u + v*v;

    float inner_feq = 1 + cur_c_dot_u/cs2 + pow((float) cur_c_dot_u, (float) 2)/two_cs4 - velocity_squared/two_cs2;

    float new_feq =  cur_w*rho*inner_feq;

    feq_global[index_position] = new_feq;
}