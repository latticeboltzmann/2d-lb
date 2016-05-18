
__kernel void
update_feq_diffusion(__global __write_only float *feq_global,
           __global __read_only float *rho_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __local float *local_u,
           __local float *local_v,
           __local float *local_rho,
           __constant float *w,
           __constant int *cx,
           __constant int *cy,
           const float cs,
           const int nx, const int ny)
{
    //Input should be a 3d workgroup.
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    const int two_d_index = y*nx + x;
    const int three_d_index = jump_id*nx*ny + two_d_index;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lz = get_local_id(2);

    const int LS0 = get_local_size(0);

    const int buf_index = LS0 * ly + lx;

    // u, v, and rho are 2d buffers. We don't want to read from them more than we have to.
    // So, we read them into local memory once as appropriate.
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lz == 0) && (x < nx) && (y < ny) && (jump_id < 9)){
        local_u[buf_index] = u_global[two_d_index];
        local_v[buf_index] = v_global[two_d_index];
        local_rho[buf_index] = rho_global[two_d_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x < nx) && (y < ny) && (jump_id < 9)){

        float u = local_u[buf_index];
        float v = local_v[buf_index];
        float rho = local_rho[buf_index];

        float cur_w = w[jump_id];
        int cur_cx = cx[jump_id];
        int cur_cy = cy[jump_id];

        float cur_c_dot_u = cur_cx*u + cur_cy*v;

        float new_feq = cur_w*rho*(1.f + cur_c_dot_u/(cs*cs));

        feq_global[three_d_index] = new_feq;
    }
}


__kernel void
update_hydro_diffusion(__global float *f_global,
             __global float *u_global,
             __global float *v_global,
             __global float *rho_global,
             const int nx, const int ny)
{
    // Assumes that u and v are imposed.
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

        //This *MUST* be run after move_bc's, as that takes care of BC's
        rho_global[two_d_index] = f0+f1+f2+f3+f4+f5+f6+f7+f8;
    }
}

__kernel void
collide_particles(__global float *f_global,
                  __global float *feq_global,
                  const float omega,
                  const int nx, const int ny)
{
    //Input should be a 3d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny) && (jump_id < 9)){
        int three_d_index = jump_id*nx*ny + y*nx + x;

        float f = f_global[three_d_index];
        float feq = feq_global[three_d_index];

        f_global[three_d_index] = f*(1-omega) + omega*feq;
    }
}

__kernel void
collide_particles_fisher(__global float *f_global,
                         __global float *feq_global,
                         const float omega, const float G,
                         __constant float *w,
                         const int nx, const int ny)
{
    //Input should be a 3d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int jump_id = get_global_id(2);

    if ((x < nx) && (y < ny) && (jump_id < 9)){
        int three_d_index = jump_id*nx*ny + y*nx + x;

        float f = f_global[three_d_index];
        float feq = feq_global[three_d_index];
        float cur_w = w[jump_id];

        f_global[three_d_index] = f*(1-omega) + omega*feq + cur_w*G*f*(1-f);
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

// ############ Obstacle Code ################

__kernel void
set_zero_velocity_in_obstacle(
    __global int *obstacle_mask,
    __global float *u_global,
    __global float *v_global,
    const int nx, const int ny)
{
    // Input should be a 2d workgroup.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;

        if (obstacle_mask[two_d_index] ==  1){
            u_global[two_d_index] = 0;
            v_global[two_d_index] = 0;
        }
    }
}

__kernel void
bounceback_in_obstacle(
    __global int *obstacle_mask,
    __global float *f_global,
    const int nx, const int ny)
{
    // Input should be a 2d workgroup.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        if (obstacle_mask[two_d_index] == 1){ // Bounce back on the obstacle
            float f1 = f_global[1*ny*nx + two_d_index];
            float f2 = f_global[2*ny*nx + two_d_index];
            float f3 = f_global[3*ny*nx + two_d_index];
            float f4 = f_global[4*ny*nx + two_d_index];
            float f5 = f_global[5*ny*nx + two_d_index];
            float f6 = f_global[6*ny*nx + two_d_index];
            float f7 = f_global[7*ny*nx + two_d_index];
            float f8 = f_global[8*ny*nx + two_d_index];

            // Bounce back everywhere!

            // Coalesce reads
            f_global[1*ny*nx + two_d_index] = f3;
            f_global[2*ny*nx + two_d_index] = f4;
            f_global[3*ny*nx + two_d_index] = f1;
            f_global[4*ny*nx + two_d_index] = f2;
            f_global[5*ny*nx + two_d_index] = f7;
            f_global[6*ny*nx + two_d_index] = f8;
            f_global[7*ny*nx + two_d_index] = f5;
            f_global[8*ny*nx + two_d_index] = f6;
        }
    }
}

