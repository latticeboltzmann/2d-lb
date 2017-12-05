#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void
update_feq_pourous(__global __write_only double *feq_global,
           __global __read_only double *rho_global,
           __global __read_only double *u_global,
           __global __read_only double *v_global,
           const double epsilon,
           __constant double *w_arr,
           __constant int *cx_arr,
           __constant int *cy_arr,
           const double cs,
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

        double rho = rho_global[three_d_index];
        const double u = u_global[three_d_index];
        const double v = v_global[three_d_index];

        // Now loop over every jumper
        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*nx*ny + three_d_index;

            double w = w_arr[jump_id];
            int cx = cx_arr[jump_id];
            int cy = cy_arr[jump_id];

            double c_dot_u = cx*u + cy*v;
            double u_squared = u*u + v*v;

            double new_feq =
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
    __global double *f_global,
    __global __read_only double *feq_global,
    __global __read_only double *rho_global,
    __global __read_only double *u_global,
    __global __read_only double *v_global,
    __global __read_only double *Fx_global,
    __global __read_only double *Fy_global,
    const double epsilon,
    const double omega,
    __constant double *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers,
    const double delta_t,
    const double cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        const double rho = rho_global[three_d_index];
        const double u = u_global[three_d_index];
        const double v = v_global[three_d_index];
        const double Fx = Fx_global[two_d_index];
        const double Fy = Fy_global[two_d_index];


        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            double relax = f_global[four_d_index]*(1-omega) + omega*feq_global[four_d_index];
            //Calculate Fi
            double c_dot_F = cx_arr[jump_id] * Fx + cy_arr[jump_id] * Fy;
            double c_dot_u = cx_arr[jump_id] * u  + cy_arr[jump_id] * v;
            double u_dot_F = Fx * u + Fy * v;

            double Fi = w_arr[jump_id]*rho*(1 - .5*omega)*(
                c_dot_F/(cs*cs)
                + c_dot_F*c_dot_u/(cs*cs*cs*cs*epsilon)
                - u_dot_F/(cs*cs*epsilon)
            );

            f_global[four_d_index] = relax + delta_t * Fi;
        }
    }
}

__kernel void
add_eating_collision(
    const int eater_index,
    const int eatee_index,
    const double eat_rate,
    __global double *f_global,
    __global __read_only double *rho_global,
    __constant double *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int num_populations,
    const int num_jumpers,
    const double delta_t,
    const double cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_eater_index = eater_index*ny*nx + two_d_index;
        int three_d_eatee_index = eatee_index*ny*nx + two_d_index;

        const double rho_eater = rho_global[three_d_eater_index];
        const double rho_eatee = rho_global[three_d_eatee_index];

        const double all_growth = delta_t * eat_rate*rho_eater*rho_eatee;

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_eater_index = jump_id*num_populations*ny*nx + three_d_eater_index;
            int four_d_eatee_index = jump_id*num_populations*ny*nx + three_d_eatee_index;

            float w = w_arr[jump_id];

            f_global[four_d_eater_index] += w * all_growth;
            f_global[four_d_eatee_index] -= w * all_growth;
        }
    }
}

__kernel void
update_velocity_prime(__global double *u_prime_global,
                      __global double *v_prime_global,
                      __global __read_only double *rho_global,
                      __global __read_only double *f_global,
                      __constant double *tau_arr,
                      __constant double *w_arr,
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

        double numerator_u = 0;
        double numerator_v = 0;

        double denominator = 0; // The denominator has no vectorial nature

        for(int cur_field=0; cur_field < num_populations; cur_field++){
            int three_d_index = cur_field*ny*nx + two_d_index;

            double cur_tau = tau_arr[cur_field];
            double rho = rho_global[three_d_index];

            for(int jump_id=0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
                double f = f_global[four_d_index];
                double cx = cx_arr[jump_id];
                double cy = cy_arr[jump_id];

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
             __global __read_only double *f_global,
             __global double *rho_global,
             __global double *u_prime_global,
             __global double *v_prime_global,
             __global double *u_global,
             __global double *v_global,
             __global __read_only double *Gx_global,
             __global __read_only double *Gy_global,
             const double epsilon,
             const double nu_fluid,
             const double Fe,
             const double K,
             __constant double *w_arr,
             __constant int *cx_arr,
             __constant int *cy_arr,
             const int nx, const int ny,
             const int cur_field,
             const int num_populations,
             const int num_jumpers,
             const double delta_t)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        double Gx = Gx_global[three_d_index];
        double Gy = Gy_global[three_d_index];

        // Update rho!
        double new_rho = 0;
        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
            double f = f_global[four_d_index];
            new_rho += f;
        }
        rho_global[three_d_index] = new_rho;
        //Now determine the new velocity
        double u_prime = u_prime_global[two_d_index];
        double v_prime = v_prime_global[two_d_index];

        //Gx NEEDS to be force/density...i.e. an acceleration, or else this just doesn't work!

        double u_temp = u_prime;
        double v_temp = v_prime;

        u_temp += (.5*delta_t*epsilon*Gx);
        v_temp += (.5*delta_t*epsilon*Gy);


        double c0 = .5*(1 + .5*epsilon*delta_t*nu_fluid/K);
        double c1 = (epsilon*.5*delta_t*Fe)/sqrt(K);

        double temp_mag = sqrt(u_temp*u_temp + v_temp*v_temp);

        double u = u_temp/(c0 + sqrt(c0*c0 + c1 * temp_mag));
        double v = v_temp/(c0 + sqrt(c0*c0 + c1 * temp_mag));

        u_global[three_d_index] = u;
        v_global[three_d_index] = v;
    }
}

__kernel void
update_forces_pourous(
    __global double *u_global,
    __global double *v_global,
    __global __read_only double *Fx_global,
    __global __read_only double *Fy_global,
    __global __read_only double *Gx_global,
    __global __read_only double *Gy_global,
    const double epsilon,
    const double nu_fluid,
    const double Fe,
    const double K,
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

        double u = u_global[three_d_index];
        double v = v_global[three_d_index];

        double Gx = Gx_global[three_d_index];
        double Gy = Gy_global[three_d_index];

        // Based on the new velocity, determine the force.
        // Note that you have to calculate the new velocity first in this formalism!
        double Fx = 0;
        double Fy = 0;

        Fx += -(epsilon * nu_fluid*u)/K;
        Fy += -(epsilon * nu_fluid*v)/K;

        double vel_mag = sqrt(u*u + v*v);

        Fx += -(epsilon * Fe * vel_mag * u)/sqrt(K);
        Fy += -(epsilon * Fe * vel_mag * v)/sqrt(K);

        Fx += epsilon*Gx;
        Fy += epsilon*Gy;

        Fx_global[two_d_index] = Fx;
        Fy_global[two_d_index] = Fy;
    }
}

__kernel void
move_periodic(__global __read_only double *f_global,
              __global __write_only double *f_streamed_global,
              __constant int *cx,
              __constant int *cy,
              const int nx, const int ny,
              const int cur_field,
              const int num_populations,
              const int num_jumpers)
{
    /* Moves you assuming periodic BC's. */
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
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
move(
    __global __read_only double *f_global,
    __global __write_only double *f_streamed_global,
    __constant int *cx,
    __constant int *cy,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];

            int stream_x = x + cur_cx;
            int stream_y = y + cur_cy;

            // Check if you are in bounds

            if ((stream_x >= 0)&&(stream_x < nx)&&(stream_y>=0)&&(stream_y<ny)){
                int slice = jump_id*num_populations*nx*ny + cur_field*nx*ny;
                int old_4d_index = slice + y*nx + x;
                int new_4d_index = slice + stream_y*nx + stream_x;

                f_streamed_global[new_4d_index] = f_global[old_4d_index];
            }
        }
    }
}



__kernel void
move_open_bcs(
    __global __read_only double *f_global,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){ // Make sure you are in the domain

        //LEFT WALL: ZERO GRADIENT, no corners
        if ((x==0) && (y >= 1)&&(y < ny-1)){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //RIGHT WALL: ZERO GRADIENT, no corners
        else if ((x==nx - 1) && (y >= 1)&&(y < ny-1)){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //We need a barrier here! The top piece must run before the bottom one...

        //TOP WALL: ZERO GRADIENT, no corners
        else if ((y == ny - 1)&&((x >= 1)&&(x < nx-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM WALL: ZERO GRADIENT, no corners
        else if ((y == 0)&&((x >= 1)&&(x < nx-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM LEFT CORNER
        else if ((x == 0)&&((y == 0))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //TOP LEFT CORNER
        else if ((x == 0)&&((y == ny-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM RIGHT CORNER
        else if ((x == nx - 1)&&((y == 0))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //TOP RIGHT CORNER
        else if ((x == nx - 1)&&((y == ny - 1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }
    }
}


__kernel void
copy_streamed_onto_f(
    __global __write_only double *f_streamed_global,
    __global __read_only double *f_global,
    __constant int *cx,
    __constant int *cy,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers)
{
    /* Moves you assuming periodic BC's. */
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cy = cy[jump_id];

            int four_d_index = jump_id*num_populations*nx*ny + cur_field*nx*ny + y*nx + x;

            f_global[four_d_index] = f_streamed_global[four_d_index];
        }
    }
}

__kernel void
add_constant_body_force(
    const int field_num,
    const double force_x,
    const double force_y,
    __global double *Gx_global,
    __global double *Gy_global,
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int three_d_index = field_num*nx*ny + y*nx + x;

        Gx_global[three_d_index] += force_x;
        Gy_global[three_d_index] += force_y;

    }
}

__kernel void
add_radial_body_force(
    const int field_num,
    const int center_x,
    const int center_y,
    const double prefactor,
    const double radial_scaling,
    __global double *Gx_global,
    __global double *Gy_global,
    const int nx, const int ny,
    const double delta_x_sim
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int three_d_index = field_num*nx*ny + y*nx + x;

        // Get the current radius and angle

        const double delta_x = x - center_x;
        const double delta_y = y - center_y;

        const double radius_dim = delta_x_sim*sqrt(delta_x*delta_x + delta_y*delta_y);
        const double theta = atan2(delta_y, delta_x);

        // Get the unit vector
        const double rhat_x = cos(theta);
        const double rhat_y = sin(theta);

        // Get the force
        double magnitude = prefactor*((double)pow(radius_dim, radial_scaling));
        Gx_global[three_d_index] += magnitude*rhat_x;
        Gy_global[three_d_index] += magnitude*rhat_y;
    }
}


__kernel void
add_interaction_force(
    const int fluid_index_1,
    const int fluid_index_2,
    const double G_int,
    __local double *local_fluid_1,
    __local double *local_fluid_2,
    __global __read_only double *rho_global,
    __global double *Gx_global,
    __global double *Gy_global,
    const double cs,
    __constant int *cx,
    __constant int *cy,
    __constant double *w,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int num_jumpers,
    const double delta_x,
    const int is_periodic,
    const int is_zero_gradient)
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
            if (is_periodic == 1){
                if (temp_x >= nx) temp_x -= nx;
                if (temp_x < 0) temp_x += nx;

                if (temp_y >= ny) temp_y -= ny;
                if (temp_y < 0) temp_y += ny;
            }
            if (is_zero_gradient == 1){
                if (temp_x >= nx) temp_x = nx - 1;
                if (temp_x < 0) temp_x = 0;

                if (temp_y >= ny) temp_y = ny - 1;
                if (temp_y < 0) temp_y = 0;
            }

            local_fluid_1[row*buf_nx + idx_1D] = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
            local_fluid_2[row*buf_nx + idx_1D] = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        //Remember, this is force PER DENSITY to avoid problems
        double force_x_fluid_1 = 0;
        double force_y_fluid_1 = 0;

        double force_x_fluid_2 = 0;
        double force_y_fluid_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            double cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            force_x_fluid_1 += cur_w * cur_cx * local_fluid_2[new_2d_buf_index];
            force_y_fluid_1 += cur_w * cur_cy * local_fluid_2[new_2d_buf_index];

            force_x_fluid_2 += cur_w * cur_cx * local_fluid_1[new_2d_buf_index];
            force_y_fluid_2 += cur_w * cur_cy * local_fluid_1[new_2d_buf_index];
        }

        force_x_fluid_1 *= -G_int/delta_x; // This is a gradient; need delta_x!
        force_y_fluid_1 *= -G_int/delta_x;

        force_x_fluid_2 *= -G_int/delta_x;
        force_y_fluid_2 *= -G_int/delta_x;

        const int two_d_index = y*nx + x;

        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}

__kernel void
add_interaction_force_second_belt(
    const int fluid_index_1,
    const int fluid_index_2,
    const double G_int,
    __local double *local_fluid_1,
    __local double *local_fluid_2,
    __global __read_only double *rho_global,
    __global double *Gx_global,
    __global double *Gy_global,
    const double cs,
    __constant int *cx,
    __constant int *cy,
    __constant double *w,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int num_jumpers,
    const double delta_x,
    const int is_periodic,
    const int is_zero_gradient)
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
            if (is_periodic == 1){
                if (temp_x >= nx) temp_x -= nx;
                if (temp_x < 0) temp_x += nx;

                if (temp_y >= ny) temp_y -= ny;
                if (temp_y < 0) temp_y += ny;
            }
            if (is_zero_gradient == 1){
                if (temp_x >= nx) temp_x = nx - 1;
                if (temp_x < 0) temp_x = 0;

                if (temp_y >= ny) temp_y = ny - 1;
                if (temp_y < 0) temp_y = 0;
            }

            local_fluid_1[row*buf_nx + idx_1D] = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
            local_fluid_2[row*buf_nx + idx_1D] = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        //Remember, this is force PER DENSITY to avoid problems
        double force_x_fluid_1 = 0;
        double force_y_fluid_1 = 0;

        double force_x_fluid_2 = 0;
        double force_y_fluid_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            double cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            force_x_fluid_1 += cur_w * cur_cx * local_fluid_2[new_2d_buf_index];
            force_y_fluid_1 += cur_w * cur_cy * local_fluid_2[new_2d_buf_index];

            force_x_fluid_2 += cur_w * cur_cx * local_fluid_1[new_2d_buf_index];
            force_y_fluid_2 += cur_w * cur_cy * local_fluid_1[new_2d_buf_index];
        }

        force_x_fluid_1 *= -G_int/delta_x; // This is a gradient; need delta_x!
        force_y_fluid_1 *= -G_int/delta_x;

        force_x_fluid_2 *= -G_int/delta_x;
        force_y_fluid_2 *= -G_int/delta_x;

        const int two_d_index = y*nx + x;

        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}
