__kernel void
update_feq_pourous(__global __write_only float *feq_global,
           __global __read_only float *rho_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           const float epsilon,
           __constant float *w_arr,
           __constant int *cx_arr,
           __constant int *cy_arr,
           const float cs,
           const int nx, const int ny,
           const int field_num,
           const int num_populations,
           const int num_jumpers)
{
    //Input should be a 2d workgroup. But, we loop over a 4d array...
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int two_d_index = y*nx + x;

    if ((x < nx) && (y < ny)){

        int three_d_index = field_num*nx*ny + two_d_index;

        float rho = rho_global[three_d_index];
        const float u = u_global[three_d_index];
        const float v = v_global[three_d_index];

        // Now loop over every jumper
        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*nx*ny + three_d_index;

            float w = w_arr[jump_id];
            int cx = cx_arr[jump_id];
            int cy = cy_arr[jump_id];

            float c_dot_u = cx*u + cy*v;
            float u_squared = u*u + v*v;

            float new_feq =
            w*rho*(
            1.f
            + c_dot_u/(cs*cs)
            + (c_dot_u*c_dot_u)/(2*cs*cs*cs*cs*epsilon)
            - u_squared/(2*cs*cs*epsilon)
            );

            feq_global[four_d_index] = new_feq;
        }
    }
}

__kernel void
collide_particles_pourous(
    __global float *f_global,
    __global __read_only float *feq_global,
    __global __read_only float *rho_global,
    __global __read_only float *u_global,
    __global __read_only float *v_global,
    __global __read_only float *Fx_global,
    __global __read_only float *Fy_global,
    const float epsilon,
    const float omega,
    __constant float *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers,
    const float delta_t,
    const float cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        float rho = rho_global[three_d_index];
        float u = u_global[three_d_index];
        float v = v_global[three_d_index];
        float Fx = Fx_global[two_d_index];
        float Fy = Fy_global[two_d_index];


        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            float f = f_global[four_d_index];
            float feq = feq_global[four_d_index];
            float w = w_arr[jump_id];
            int cx = cx_arr[jump_id];
            int cy = cy_arr[jump_id];

            float relax = f*(1-omega) + omega*feq;
            //Calculate Fi
            float c_dot_F = cx * Fx + cy * Fy;
            float c_dot_u = cx * u  + cy * v;
            float u_dot_F = Fx * u + Fy * v;

            float Fi = w*rho*(1 - .5*omega)*(
                1.
                + c_dot_F/(cs*cs)
                + c_dot_F*c_dot_u/(cs*cs*cs*cs*epsilon)
                - u_dot_F/(cs*cs*epsilon)
            );

            float new_f = relax + delta_t * Fi;

            f_global[four_d_index] = new_f;
        }
    }
}

__kernel void
update_velocity_prime(__global float *u_prime_global,
                      __global float *v_prime_global,
                      __global __read_only float *rho_global,
                      __global __read_only float *f_global,
                      __constant float *tau_arr,
                      __constant float *w_arr,
                      __constant int *cx_arr,
                      __constant int *cy_arr,
                      const int nx, const int ny,
                      const int num_populations,
                      const int num_jumpers
                      )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;

        float numerator_u = 0;
        float numerator_v = 0;

        float denominator = 0; // The denominator has no vectorial nature

        for(int cur_field=0; cur_field < num_populations; cur_field++){
            int three_d_index = cur_field*ny*nx + two_d_index;

            float cur_tau = tau_arr[cur_field];
            float rho = rho_global[three_d_index];

            for(int jump_id=0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
                float f = f_global[four_d_index];
                float cx = cx_arr[jump_id];
                float cy = cy_arr[jump_id];

                numerator_u += cx * f;
                numerator_v += cy * f;
            }
            numerator_u /= cur_tau;
            numerator_v /= cur_tau;

            denominator += rho/cur_tau;
        }
    u_prime_global[two_d_index] = numerator_u/denominator;
    v_prime_global[two_d_index] = numerator_v/denominator;
    }
}

__kernel void
update_hydro_pourous(
             __global __read_only float *f_global,
             __global float *rho_global,
             __global float *u_prime_global,
             __global float *v_prime_global,
             __global float *u_global,
             __global float *v_global,
             __global __read_only float *Gx_global,
             __global __read_only float *Gy_global,
             const float epsilon,
             const float nu_fluid,
             const float Fe,
             const float K,
             __constant float *w_arr,
             __constant int *cx_arr,
             __constant int *cy_arr,
             const int nx, const int ny,
             const int cur_field,
             const int num_populations,
             const int num_jumpers,
             const float delta_t)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        float Gx = Gx_global[three_d_index];
        float Gy = Gy_global[three_d_index];

        // Update rho!
        float new_rho = 0;
        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
            float f = f_global[four_d_index];
            new_rho += f;
        }
        rho_global[three_d_index] = new_rho;
        //Now determine the new velocity
        float u_prime = u_prime_global[two_d_index];
        float v_prime = v_prime_global[two_d_index];

        float u_temp = u_prime + (.5*delta_t*epsilon*Gx)/(new_rho);
        float v_temp = v_prime + (.5*delta_t*epsilon*Gy)/new_rho;

        float c0 = .5*(1 + .5*epsilon*delta_t*nu_fluid/K);
        float c1 = (epsilon*.5*delta_t*Fe)/sqrt(K);

        float temp_mag = sqrt(u_temp*u_temp + v_temp*v_temp);

        float u = u_temp/(c0 + sqrt(c0*c0 + c1 * temp_mag));
        float v = v_temp/(c0 + sqrt(c0*c0 + c1 * temp_mag));

        u_global[three_d_index] = u;
        v_global[three_d_index] = v;
    }
}

__kernel void
update_forces_pourous(
    __global float *u_global,
    __global float *v_global,
    __global __read_only float *Fx_global,
    __global __read_only float *Fy_global,
    __global __read_only float *Gx_global,
    __global __read_only float *Gy_global,
    const float epsilon,
    const float nu_fluid,
    const float Fe,
    const float K,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        float u = u_global[three_d_index];
        float v = v_global[three_d_index];

        float Gx = Gx_global[three_d_index];
        float Gy = Gy_global[three_d_index];

        // Based on the new velocity, determine the force.
        // Note that you have to calculate the new velocity first in this formalism!
        float Fx = 0;
        float Fy = 0;

        Fx += -(epsilon * nu_fluid*u)/K;
        Fy += -(epsilon * nu_fluid*v)/K;

        float vel_mag = sqrt(u*u + v*v);

        Fx += -(epsilon * Fe * vel_mag * u)/sqrt(K);
        Fy += -(epsilon * Fe * vel_mag * v)/sqrt(K);

        Fx += epsilon*Gx;
        Fy += epsilon*Gy;

        Fx_global[two_d_index] = Fx;
        Fy_global[two_d_index] = Fy;
    }
}

__kernel void
move_periodic(__global __read_only float *f_global,
              __global __write_only float *f_streamed_global,
              __constant int *cx,
              __constant int *cy,
              const int nx, const int ny,
              const int cur_field,
              const int num_populations)
{
    /* Moves you assuming periodic BC's. */
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < 9; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];

            //Make sure that you don't go out of the system

            int stream_x = x + cur_cx;
            int stream_y = y + cur_cy;

            if (stream_x == nx) stream_x = 0;
            if (stream_x < 0) stream_x = nx - 1;

            if (stream_y == ny) stream_y = 0;
            if (stream_y < 0) stream_y = ny - 1;

            int slice = jump_id*num_populations*nx*ny + cur_field*nx*ny;
            int old_4d_index = slice + y*nx + x;
            int new_4d_index = slice + stream_y*nx + stream_x;

            f_streamed_global[new_4d_index] = f_global[old_4d_index];
        }
    }
}

__kernel void
update_surf_tension(__global float *S_global,
                __global __read_only float *rho_global,
                const float c_o, const float alpha,
                const int nx, const int ny, const int surf_index)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = surf_index*ny*nx + two_d_index;

        float cur_rho = rho_global[three_d_index];
        if (cur_rho < 0) cur_rho = 0;
        S_global[two_d_index] = pow(1.f - exp(-cur_rho/c_o), alpha);
    }
}

__kernel void
update_pressure_force(__global __read_only float *rho_global,
                    const int pop_index,
                    __global float *pseudo_force_x,
                    __global float *pseudo_force_y,
                    const float G_chen,
                    const float rho_o,
                    const float cs,
                    __constant int *cx,
                    __constant int *cy,
                    __constant float *w,
                    __local float *rho_local,
                    const int nx, const int ny,
                    const int buf_nx, const int buf_ny,
                    const int halo)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            if (temp_x >= nx) temp_x -= nx;
            if (temp_x < 0) temp_x += nx;

            if (temp_y >= ny) temp_y -= ny;
            if (temp_y < 0) temp_y += ny;

            rho_local[row*buf_nx + idx_1D] = rho_global[pop_index*ny*nx + temp_y*nx + temp_x];

        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired psi are read in, do the multiplication
    if ((x < nx) && (y < ny)){
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;
        const float middle_rho = rho_local[old_2d_buf_index];

        float rho_grad_x = 0;
        float rho_grad_y = 0;
        for(int jump_id = 0; jump_id < 9; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            float cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            float new_rho = rho_local[new_2d_buf_index];
            rho_grad_x += cur_w * cur_cx * new_rho;
            rho_grad_y += cur_w * cur_cy * new_rho;
        }
        const int two_d_index = y*nx + x;

        const float inv_cs_sq = 1./(cs*cs);
        rho_grad_x *= inv_cs_sq; // To make the gradient actually correct
        rho_grad_y *= inv_cs_sq;

        //Now calculate the pressure gradient

        pseudo_force_x[two_d_index] = -G_chen * rho_grad_x * (middle_rho - rho_o);
        pseudo_force_y[two_d_index] = -G_chen * rho_grad_y * (middle_rho - rho_o);
    }
}

__kernel void
update_surface_forces(__global __read_only float *S_global,
               __global __write_only float *surface_force_x,
               __global __write_only float *surface_force_y,
               const int surf_index,
               const float cs,
               const float epsilon,
               __constant int *cx,
               __constant int *cy,
               __constant float *w,
               __local float *S_local,
               const int nx, const int ny,
               const int buf_nx, const int buf_ny,
               const int halo)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            if (temp_x >= nx) temp_x -= nx;
            if (temp_x < 0) temp_x += nx;

            if (temp_y >= ny) temp_y -= ny;
            if (temp_y < 0) temp_y += ny;

            S_local[row*buf_nx + idx_1D] = S_global[temp_y*nx + temp_x];

        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired psi are read in, do the multiplication
    if ((x < nx) && (y < ny)){
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        float grad_x = 0;
        float grad_y = 0;
        for(int jump_id = 0; jump_id < 9; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            float cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            float cur_S = S_local[new_2d_buf_index];
            grad_x += cur_w * cur_cx * cur_S;
            grad_y += cur_w * cur_cy * cur_S;
        }
        const int two_d_index = y*nx + x;
        const float inv_cs_sq = 1./(cs*cs);
        surface_force_x[two_d_index] = -epsilon * inv_cs_sq * grad_x;
        surface_force_y[two_d_index] = -epsilon * inv_cs_sq * grad_y;
    }
}

