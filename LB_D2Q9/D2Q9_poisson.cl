__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *rho_global,
           __constant float *w,
           const int nx, const int ny)
{
    //Input should be a 2d workgroup.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int two_d_index = y*nx + x;

    if ((x < nx) && (y < ny)){

        const float rho = rho_global[two_d_index];

        // 0 is handled separately for whatever reason...
        int three_d_index = 0*nx*ny + two_d_index;
        float cur_w = w[0];
        float new_feq = (cur_w - 1.0) * rho;

        feq_global[three_d_index] = new_feq;

        for(int jump_id=1; jump_id < 9; jump_id++){
            int three_d_index = jump_id*nx*ny + two_d_index;
            float cur_w = w[jump_id];
            float new_feq = cur_w*rho;
            feq_global[three_d_index] = new_feq;
        }
    }
}


__kernel void
update_hydro(__global float *f_global,
             __global float *rho_global,
             const int nx, const int ny)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int two_d_index = y*nx + x;


        float old_rho = rho_global[two_d_index];
        // No need for f0 in this formalism...
        // float f0 = f_global[0*ny*nx + two_d_index];
        float f1 = f_global[1*ny*nx + two_d_index];
        float f2 = f_global[2*ny*nx + two_d_index];
        float f3 = f_global[3*ny*nx + two_d_index];
        float f4 = f_global[4*ny*nx + two_d_index];
        float f5 = f_global[5*ny*nx + two_d_index];
        float f6 = f_global[6*ny*nx + two_d_index];
        float f7 = f_global[7*ny*nx + two_d_index];
        float f8 = f_global[8*ny*nx + two_d_index];

        float new_rho = (9./5.)*(f1+f2+f3+f4+f5+f6+f7+f8);

        rho_global[two_d_index] = new_rho;
    }
}

__kernel void
collide_particles(__global float *f_global,
                  __global float *feq_global,
                  __global float *sources,
                  const float omega,
                  __constant float *w,
                  const float delta_t, const float D,
                  const int nx, const int ny)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){

        const int two_d_index = y*nx + x;

        // Define the reaction term
        const float react = sources[two_d_index] * delta_t * D;

        for(int jump_id = 0; jump_id < 9; jump_id++){
            int three_d_index = jump_id*nx*ny + two_d_index;

            float f = f_global[three_d_index];
            float feq = feq_global[three_d_index];
            float cur_w = w[jump_id];

            float new_f = f*(1-omega) + omega*feq + cur_w*react;

            f_global[three_d_index] = new_f;
        }
    }
}

__kernel void
copy_buffer(__global __read_only float *copy_from,
            __global __write_only float *copy_to,
            const int nx, const int ny)
{
    //Assumes a 3d workgroup
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny) && (jump_id < 9)){
        int three_d_index = jump_id*nx*ny + y*nx + x;
        copy_to[three_d_index] = copy_from[three_d_index];
    }
}

__kernel void
move(__global __read_only float *f_global,
     __global __write_only float *f_streamed_global,
     __constant int *cx,
     __constant int *cy,
     const int nx, const int ny)
{
    //Input should be a 3d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny) && (jump_id < 9)){
        //Only stream if you will not go out of the system.

        int cur_cx = cx[jump_id];
        int cur_cy = cy[jump_id];

        //Make sure that you don't go out of the system

        int stream_x = x + cur_cx;
        int stream_y = y + cur_cy;

        const int old_3d_index = jump_id*nx*ny + y*nx + x;

        if ((stream_x >= 0)&&(stream_x < nx)&&(stream_y>=0)&&(stream_y<ny)){ // Stream
            const int new_3d_index = jump_id*nx*ny + stream_y*nx + stream_x;
            //Need two buffers to avoid parallel updates & shennanigans.
            f_streamed_global[new_3d_index] = f_global[old_3d_index];
        }
        //TODO: See if we can avoid copying later and avoid bizarre movement problems
    }
}

__kernel void
move_bcs(__global float *f_global,
         const float rho_specified,
         __constant float *w,
         const int nx, const int ny)
{
    //TODO: Make this efficient. I recognize there are better ways to do this, perhaps a kernel for each boundary...
    //Input should be a 2d workgroup! Everything is done inplace, no need for a second buffer
    //Must be run after move
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int two_d_index = y*nx + x;

    bool on_left = (x==0) && (y >= 1)&&(y < ny-1);
    bool on_right = (x==nx - 1) && (y >= 1)&&(y < ny -1);
    bool on_top = (y == ny-1) && (x >= 1) && (x< nx-1);
    bool on_bottom = (y == 0) && (x >= 1) && (x < nx-1);

    bool on_main_surface = on_left || on_right || on_top || on_bottom;

    bool bottom_left_corner = (x==0) && (y==0);
    bool bottom_right_corner = (x==nx-1)&&(y==0);
    bool upper_left_corner = (x==0)&&(y==ny-1);
    bool upper_right_corner = (x==nx-1)&&(y==ny-1);

    bool on_corner = bottom_left_corner || bottom_right_corner || upper_left_corner || upper_right_corner;

    if (on_main_surface || on_corner){

        // You need to read in all f...except f0 actually
        // float f0 = f_global[0*ny*nx + two_d_index];
        float f1 = f_global[1*ny*nx + two_d_index];
        float f2 = f_global[2*ny*nx + two_d_index];
        float f3 = f_global[3*ny*nx + two_d_index];
        float f4 = f_global[4*ny*nx + two_d_index];
        float f5 = f_global[5*ny*nx + two_d_index];
        float f6 = f_global[6*ny*nx + two_d_index];
        float f7 = f_global[7*ny*nx + two_d_index];
        float f8 = f_global[8*ny*nx + two_d_index];

        //Top: Constant density
        if (on_top){
            float rho_to_add = -(f1 + f2 + f3 + f5 + f6 + (-1 + w[0])*rho_specified)/(w[4]+w[7]+w[8]);
            f_global[7*ny*nx + two_d_index] = w[7] * rho_to_add;
            f_global[4*ny*nx + two_d_index] = w[4] * rho_to_add;
            f_global[8*ny*nx + two_d_index] = w[8] * rho_to_add;
        }
        //Right: constant density
        if (on_right){
            float rho_to_add = -(f1 + f2 + f4 + f5 + f8 + (-1 + w[0])*rho_specified)/(w[3]+w[6]+w[7]);
            f_global[3*ny*nx + two_d_index] = w[3] * rho_to_add;
            f_global[6*ny*nx + two_d_index] = w[6] * rho_to_add;
            f_global[7*ny*nx + two_d_index] = w[7] * rho_to_add;
        }

        //Bottom : constant density
        if (on_bottom){
            float rho_to_add = -(f1 + f3 + f4 + f7 + f8 + (-1 + w[0])*rho_specified)/(w[2]+w[5]+w[6]);
            f_global[2*ny*nx + two_d_index] = w[2] * rho_to_add;
            f_global[5*ny*nx + two_d_index] = w[5] * rho_to_add;
            f_global[6*ny*nx + two_d_index] = w[6] * rho_to_add;
        }
        //Left: Constant density
        if (on_left){
            float rho_to_add = -(f2 + f3 + f4 + f6 + f7 + (-1 + w[0])*rho_specified)/(w[1]+w[5]+w[8]);
            f_global[1*ny*nx + two_d_index] = w[1] * rho_to_add;
            f_global[5*ny*nx + two_d_index] = w[5] * rho_to_add;
            f_global[8*ny*nx + two_d_index] = w[8] * rho_to_add;
        }

        //Corner nodes! Extremely annoying and painful, and likely slow

        //Bottom left corner: Constant density
        if (bottom_left_corner){
            float rho_to_add = -(f3 + f4 + f6 + f7 + f8 + (-1 + w[0])*rho_specified)/(w[1]+w[2]+w[5]);
            f_global[1*ny*nx + two_d_index] = w[1] * rho_to_add;
            f_global[2*ny*nx + two_d_index] = w[2] * rho_to_add;
            f_global[5*ny*nx + two_d_index] = w[5] * rho_to_add;
        }

        // Bottom right corner: Constant density
        if (bottom_right_corner){
            float rho_to_add = -(f1 + f4 + f5 + f7 + f8 + (-1 + w[0])*rho_specified)/(w[2]+w[3]+w[6]);
            f_global[2*ny*nx + two_d_index] = w[2] * rho_to_add;
            f_global[3*ny*nx + two_d_index] = w[3] * rho_to_add;
            f_global[6*ny*nx + two_d_index] = w[6] * rho_to_add;
        }

        // Upper left corner: constant density
        if (upper_left_corner){
            float rho_to_add = -(f2 + f3 + f5 + f6 + f7 + (-1 + w[0])*rho_specified)/(w[1]+w[4]+w[8]);
            f_global[1*ny*nx + two_d_index] = w[1] * rho_to_add;
            f_global[4*ny*nx + two_d_index] = w[4] * rho_to_add;
            f_global[8*ny*nx + two_d_index] = w[8] * rho_to_add;
        }

        // Upper right corner: constant density
        if (upper_right_corner){
            float rho_to_add = -(f1 + f2 + f5 + f6 + f8 + (-1 + w[0])*rho_specified)/(w[3]+w[4]+w[7]);
            f_global[3*ny*nx + two_d_index] = w[3] * rho_to_add;
            f_global[4*ny*nx + two_d_index] = w[4] * rho_to_add;
            f_global[7*ny*nx + two_d_index] = w[7] * rho_to_add;
        }
    }
}

__kernel void
get_gradient(__global __read_only float *rho,
     __global __write_only float *u,
     __global __write_only float *v,
     const int nx, const int ny)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int two_d_index = y*nx + x;

    float up = y + 1;
    float down = y - 1;
    float right = x + 1;
    float left = x - 1;

    if ((x < nx) && (y < ny) && (jump_id < 9)){
        //Only stream if you will not go out of the system.

        int cur_cx = cx[jump_id];
        int cur_cy = cy[jump_id];

        //Make sure that you don't go out of the system

        int stream_x = x + cur_cx;
        int stream_y = y + cur_cy;

        const int old_3d_index = jump_id*nx*ny + y*nx + x;

        if ((stream_x >= 0)&&(stream_x < nx)&&(stream_y>=0)&&(stream_y<ny)){ // Stream
            const int new_3d_index = jump_id*nx*ny + stream_y*nx + stream_x;
            //Need two buffers to avoid parallel updates & shennanigans.
            f_streamed_global[new_3d_index] = f_global[old_3d_index];
        }
        //TODO: See if we can avoid copying later and avoid bizarre movement problems
    }
}