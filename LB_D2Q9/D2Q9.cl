
__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __global __read_only float *rho_global,
           int nx, int ny)
{
    //Input should be a 3d workgroup.
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny) && (jump_id < 9)){

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

        //The position in 3d is confusing...REMEMBER, U, V, AND RHO ARE 2D. BUT, FEQ IS 3D!

        int two_d_index = y*nx + x;
        int three_d_index = jump_id*nx*ny + two_d_index;

        float u = u_global[two_d_index];
        float v= v_global[two_d_index];
        float rho = rho_global[two_d_index];

        // We used to have a bunch of if statements here. It's better to have something thata
        // can be executed in parallel.

        float cur_w = w[jump_id];
        int cur_cx = cx[jump_id];
        int cur_cy = cy[jump_id];

        float cur_c_dot_u = cur_cx*u + cur_cy*v;
        float velocity_squared = u*u + v*v;

        float inner_feq = 1 + cur_c_dot_u/cs2 + pow(cur_c_dot_u,2.f)/two_cs4 - velocity_squared/two_cs2;

        float new_feq =  cur_w*rho*inner_feq;

        feq_global[three_d_index] = new_feq;

    }
}


__kernel void
update_hydro(__global float *f_global,
             __global float *u_global,
             __global float *v_global,
             __global float *rho_global,
             float inlet_rho, float outlet_rho,
             int nx, int ny)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int two_d_index = y*nx + x;
        float f0 = f_global[0*ny*nx + two_d_index];
        float f1 = f_global[1*ny*nx + two_d_index];
        float f2 = f_global[2*ny*nx + two_d_index];
        float f3 = f_global[3*ny*nx + two_d_index];
        float f4 = f_global[4*ny*nx + two_d_index];
        float f5 = f_global[5*ny*nx + two_d_index];
        float f6 = f_global[6*ny*nx + two_d_index];
        float f7 = f_global[7*ny*nx + two_d_index];
        float f8 = f_global[8*ny*nx + two_d_index];

        float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8;
        rho_global[two_d_index] = rho;
        float inverse_rho = 1./rho;

        u_global[two_d_index] = (f1-f3+f5-f6-f7+f8)*inverse_rho;
        v_global[two_d_index] = (f5+f2+f6-f7-f4-f8)*inverse_rho;

        //Now do the boundary conditions. It is faster to do it here so we don't have to
        //reread variables! I think two if statements are needed...I don't see a way around it.

        if (x==0){
            rho_global[two_d_index] = inlet_rho;
            u_global[two_d_index] = 1 - (f0+f2+f4+2*(f3+f6+f7))/inlet_rho;
        }
        if (x==nx-1){
            rho_global[two_d_index] = outlet_rho;
            u_global[two_d_index] = -1 + (f0+f2+f4+2*(f1+f5+f8))/outlet_rho;
        }
    }
}

__kernel void
collide_particles(__global float *f_global,
                  __global float *feq_global,
                  float omega,
                  int nx, int ny)
{
    //Input should be a 3d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny)){
        int three_d_index = jump_id*nx*ny + y*nx + x;

        float f = f_global[three_d_index];
        float feq = feq_global[three_d_index];

        f_global[three_d_index] = f*(1.f-omega) + omega*feq;
    }
}

__kernel void
move(__global float *f_global,
     __global float *f_streamed_global,
     int nx, int ny)
{
    //Input should be a 3d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    //Only stream if you will not go out of the system.
    const int cx[9] = {0,1,0,-1,0,1,-1,-1,1}; // direction vector for the x direction
    const int cy[9] = {0,0,1,0,-1,1,1,-1,-1}; // direction vector for the y direction

    int cur_cx = cx[jump_id];
    int cur_cy = cy[jump_id];

    //Make sure that you don't go out of the system

    int stream_x = x + cur_cx;
    int stream_y = y + cur_cy;

    if ((stream_x >= 0)&&(stream_x < nx)&&(stream_y>=0)&&(stream_y<ny)){
        int old_3d_index = jump_id*nx*ny + y*nx + x;
        int new_3d_index = jump_id*nx*ny + stream_y*nx + stream_x;
        //Need two buffers to avoid parallel updates & shennanigans.
        f_streamed_global[new_3d_index] = f_global[old_3d_index];
    }
}

__kernel void
move_bcs(__global float *f_global,
     __global float *u_global,
     float inlet_rho, float outlet_rho,
     int nx, int ny)
{
    //Input should be a 2d workgroup! Everything is done inplace, no need for a second buffer
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int two_d_index = y*nx + x;

    float f0 = f_global[0*ny*nx + two_d_index];
    float f1 = f_global[1*ny*nx + two_d_index];
    float f2 = f_global[2*ny*nx + two_d_index];
    float f3 = f_global[3*ny*nx + two_d_index];
    float f4 = f_global[4*ny*nx + two_d_index];
    float f5 = f_global[5*ny*nx + two_d_index];
    float f6 = f_global[6*ny*nx + two_d_index];
    float f7 = f_global[7*ny*nx + two_d_index];
    float f8 = f_global[8*ny*nx + two_d_index];

    float u = u_global[two_d_index];

    //INLET: constant pressure
    if ((x==0) && (1<=y)&&(y < ny -1)){
        f_global[1*ny*nx + two_d_index] = f3 + (2./3.)*inlet_rho*u;
        f_global[5*ny*nx + two_d_index] = -.5*f2 +.5*f4 + f7 + (1./6.)*u*inlet_rho;
        f_global[8*ny*nx + two_d_index] = .5*f2- .5*f4 + f6 + (1./6.)*u*inlet_rho;
    }
    //OUTLET: constant pressure
    if ((x==nx - 1) && (1<=y)&&(y < ny -1)){
        f_global[3*ny*nx + two_d_index] = f1 - (2./3.)*outlet_rho*u;
        f_global[6*ny*nx + two_d_index] = -.5*f2 + .5*f4 + f8 - (1./6.)*u*outlet_rho;
        f_global[7*ny*nx + two_d_index] = .5*f2 - .5*f4 + f5 -(1./6.)*u*outlet_rho;
    }

    //NORTH: solid; bounce back
    if ((y == nx-1)&&(x <=1) && (x < nx - 1)){
        f_global[4*ny*nx + two_d_index] = f2;
        f_global[8*ny*nx + two_d_index] = f6;
        f_global[7*ny*nx + two_d_index] = f5;
    }

    //SOUTH: solid; bounce back
    if ((y == 0)&&(x <=1) && (x < nx - 1)){
        f_global[2*ny*nx + two_d_index] = f4;
        f_global[6*ny*nx + two_d_index] = f8;
        f_global[5*ny*nx + two_d_index] = f7;
    }

    //Corner nodes: tricky and a huge pain! And likely very slow.
    // BOTTOM INLET
    if ((x==0) && (y==0)){
        f_global[1*ny*nx + two_d_index] = f3;
        f_global[2*ny*nx + two_d_index] = f4;
        f_global[5*ny*nx + two_d_index] = f7;
        f_global[6*ny*nx + two_d_index] = .5*(-f0-2*f3-2*f4-2*f7+inlet_rho);
        f_global[8*ny*nx + two_d_index] = .5*(-f0-2*f3-2*f4-2*f7+inlet_rho);
    }
    // TOP INLET
    if ((x==0)&&(y==ny-1)){
        f_global[1*ny*nx + two_d_index] = f3;
        f_global[4*ny*nx + two_d_index] = f2;
        f_global[8*ny*nx + two_d_index] = f6;
        f_global[5*ny*nx + two_d_index] = .5*(-f0-2*f2-2*f3-2*f6+inlet_rho);
        f_global[7*ny*nx + two_d_index] = .5*(-f0-2*f2-2*f3-2*f6+inlet_rho);
    }

    // BOTTOM OUTLET
    if ((x==nx-1)&&(y==0)){
        f_global[3*ny*nx + two_d_index] = f1;
        f_global[2*ny*nx + two_d_index] = f4;
        f_global[6*ny*nx + two_d_index] = f8;
        f_global[5*ny*nx + two_d_index] = .5*(-f0-2*f1-2*f4-2*f8+outlet_rho);
        f_global[8*ny*nx + two_d_index] = .5*(-f0-2*f1-2*f4-2*f8+outlet_rho);
    }
    // TOP OUTLET
    if ((x==nx-1)&&(y==ny-1)){
        f_global[3*ny*nx + two_d_index] = f1;
        f_global[4*ny*nx + two_d_index] = f2;
        f_global[7*ny*nx + two_d_index] = f5;
        f_global[6*ny*nx + two_d_index] = .5*(-f0-2*f1-2*f2-2*f5+outlet_rho);
        f_global[8*ny*nx + two_d_index] = .5*(-f0-2*f1-2*f2-2*f5+outlet_rho);
    }
}

