#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#define ZERO_DENSITY 1e-6

__kernel void
update_feq_pourous(
    __global __write_only double *feq_global,
    __global __read_only double *rho_global,
    __global __read_only double *u_bary_global,
    __global __read_only double *v_bary_global,
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
        const double u = u_bary_global[two_d_index];
        const double v = v_bary_global[two_d_index];

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
    __global __read_only double *u_bary_global,
    __global __read_only double *v_bary_global,
    __global __read_only double *Gx_global,
    __global __read_only double *Gy_global,
    const double epsilon,
    const double omega,
    __constant double *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers,
    const double cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        const double rho = rho_global[three_d_index];
        const double u = u_bary_global[two_d_index];
        const double v = v_bary_global[two_d_index];
        const double Gx = Gx_global[three_d_index];
        const double Gy = Gy_global[three_d_index];


        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            double relax = f_global[four_d_index]*(1-omega) + omega*feq_global[four_d_index];
            //Calculate Fi
            double c_dot_F = cx_arr[jump_id] * Gx + cy_arr[jump_id] * Gy;
            double c_dot_u = cx_arr[jump_id] * u  + cy_arr[jump_id] * v;
            double u_dot_F = Gx * u + Gy * v;

            double Fi = w_arr[jump_id]*rho*(1 - .5*omega)*(
                c_dot_F/(cs*cs)
                + c_dot_F*c_dot_u/(cs*cs*cs*cs*epsilon)
                - u_dot_F/(cs*cs*epsilon)
            );

            f_global[four_d_index] = relax + Fi;
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

        const double all_growth = eat_rate*rho_eater*rho_eatee;

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
update_bary_velocity(
    __global double *u_bary_global,
    __global double *v_bary_global,
    __global __read_only double *rho_global,
    __global __read_only double *f_global,
    __global __read_only double *Gx_global,
    __global __read_only double *Gy_global,
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

        double sum_x = 0;
        double sum_y = 0;
        double rho_sum = 0;

        for(int cur_field=0; cur_field < num_populations; cur_field++){
            int three_d_index = cur_field*ny*nx + two_d_index;

            double cur_rho = rho_global[three_d_index];
            rho_sum += cur_rho;

            double Gx = Gx_global[three_d_index];
            double Gy = Gy_global[three_d_index];

            for(int jump_id=0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
                double f = f_global[four_d_index];
                int cx = cx_arr[jump_id];
                int cy = cy_arr[jump_id];

                sum_x += cx * f;
                sum_y += cy * f;
            }
            sum_x += cur_rho*Gx/2.;
            sum_y += cur_rho*Gy/2.;
        }
        u_bary_global[two_d_index] = sum_x/rho_sum;
        v_bary_global[two_d_index] = sum_y/rho_sum;
    }
}

__kernel void
update_hydro_pourous(
    __global __read_only double *f_global,
    __global double *rho_global,
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
    const int num_jumpers
)
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
        double new_u = 0;
        double new_v = 0;

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
            double f = f_global[four_d_index];

            new_rho += f;

            int cx = cx_arr[jump_id];
            int cy = cy_arr[jump_id];

            new_u += f*cx;
            new_v += f*cy;
        }
        rho_global[three_d_index] = new_rho;

        if(new_rho > ZERO_DENSITY){
            u_global[three_d_index] = new_u/new_rho;
            v_global[three_d_index] = new_v/new_rho;
        }
        else{
            u_global[three_d_index] = 0;
            v_global[three_d_index] = 0;
        }
    }
}

__kernel void
update_forces_pourous(
    __global double *rho_global,
    __global double *u_global,
    __global double *v_global,
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        double rho = rho_global[three_d_index];

        if (rho > ZERO_DENSITY){

            double u = u_global[three_d_index];
            double v = v_global[three_d_index];

            double Gx = Gx_global[three_d_index];
            double Gy = Gy_global[three_d_index];

            //TODO: should these be divided by density?

            //At this point, Gx and Gy are already nonzero. We must account
            //for porosity before doing anything else...

            Gx *= epsilon;
            Gy *= epsilon;

            // Now add everything else...

            Gx += -(epsilon * nu_fluid*u)/K;
            Gy += -(epsilon * nu_fluid*v)/K;

            double vel_mag = sqrt(u*u + v*v);

            Gx += -(epsilon * Fe * vel_mag * u)/sqrt(K);
            Gy += -(epsilon * Fe * vel_mag * v)/sqrt(K);

            Gx_global[three_d_index] = Gx;
            Gy_global[three_d_index] = Gy;
        }
        else{
            Gx_global[three_d_index] = 0;
            Gy_global[three_d_index] = 0;
        }
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

        // Rembmer, force PER density! In *dimensionless* units.
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
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int three_d_index = field_num*nx*ny + y*nx + x;

        // Get the current radius and angle

        const double dx = x - center_x;
        const double dy = y - center_y;

        const double radius_dim = sqrt(dx*dx + dy*dy);
        const double theta = atan2(dy, dx);

        // Get the unit vector
        const double rhat_x = cos(theta);
        const double rhat_y = sin(theta);

        // Get the force
        double magnitude = prefactor*((double)pow(radius_dim, radial_scaling));
        Gx_global[three_d_index] += magnitude*rhat_x;
        Gy_global[three_d_index] += magnitude*rhat_y;
    }
}

void get_psi(
    const int PSI_SPECIFIER,
    double rho_1, double rho_2,
    double *psi_1, double *psi_2,
    __constant double *parameters)
{
    if(PSI_SPECIFIER == 0){ // rho_1 * rho_2
        *psi_1 = rho_1;
        *psi_2 = rho_2;
    }
    if(PSI_SPECIFIER == 1){ // shan-chen
        double rho_0 = parameters[0];
        *psi_1 = rho_0*(1 - exp(-rho_1/rho_0));
        *psi_2 = rho_0*(1 - exp(-rho_2/rho_0));
    }
    if(PSI_SPECIFIER == 2){ // pow(rho_1, alpha) * pow(rho_2, alpha)
        if (rho_1 > ZERO_DENSITY){
            *psi_1 = (double)pow(rho_1, parameters[0]);
        }
        else{
            *psi_1 = 0;
        }
        if (rho_2 > ZERO_DENSITY){
            *psi_2 = (double)pow(rho_2, parameters[0]);
        }
        else{
            *psi_2 = 0;
        }
    }
    if(PSI_SPECIFIER==3){ //van-der-waals; G MUST BE SET TO ONE TO USE THIS
        double a = parameters[0];
        double b = parameters[1];
        double T = parameters[2];
        double cs = parameters[3];

        double P1 = (rho_1*T)/(1 - rho_1*b) - a*rho_1*rho_1;
        double P2 = (rho_2*T)/(1 - rho_2*b) - a*rho_2*rho_2;

        *psi_1 = sqrt(2*(P1 - cs*cs*rho_1)/(cs*cs));
        *psi_2 = sqrt(2*(P2 - cs*cs*rho_2)/(cs*cs));
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
    const int BC_SPECIFIER,
    const int PSI_SPECIFIER,
    __constant double *parameters)
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
            if (BC_SPECIFIER == 0){ //PERIODIC
                if (temp_x >= nx) temp_x -= nx;
                if (temp_x < 0) temp_x += nx;

                if (temp_y >= ny) temp_y -= ny;
                if (temp_y < 0) temp_y += ny;
            }
            if (BC_SPECIFIER == 1){ // ZERO GRADIENT
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

        //Remember, this is force PER DENSITY to avoid problems
        double force_x_fluid_1 = 0;
        double force_y_fluid_1 = 0;

        double force_x_fluid_2 = 0;
        double force_y_fluid_2 = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        double rho_1_pixel = local_fluid_1[old_2d_buf_index];
        double rho_2_pixel = local_fluid_2[old_2d_buf_index];

        double psi_1_pixel = 0;
        double psi_2_pixel = 0;

        get_psi(PSI_SPECIFIER, rho_1_pixel, rho_2_pixel, &psi_1_pixel, &psi_2_pixel, parameters);

        double psi_1 = 0; // The psi that correspond to jumping around the lattice
        double psi_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            double cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_rho_1 = local_fluid_1[new_2d_buf_index];
            double cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        force_x_fluid_1 *= -(G_int*psi_1_pixel);
        force_y_fluid_1 *= -(G_int*psi_1_pixel);

        force_x_fluid_2 *= -(G_int*psi_2_pixel);
        force_y_fluid_2 *= -(G_int*psi_2_pixel);

        const int two_d_index = y*nx + x;
        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        if(rho_1_pixel > ZERO_DENSITY){
            Gx_global[three_d_index_fluid_1] += force_x_fluid_1/rho_1_pixel;
            Gy_global[three_d_index_fluid_1] += force_y_fluid_1/rho_1_pixel;
        }
        if(rho_2_pixel > ZERO_DENSITY){
            Gx_global[three_d_index_fluid_2] += force_x_fluid_2/rho_2_pixel;
            Gy_global[three_d_index_fluid_2] += force_y_fluid_2/rho_2_pixel;
        }
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
    __constant double *pi1,
    __constant int *cx1,
    __constant int *cy1,
    const int num_jumpers_1,
    __constant double *pi2,
    __constant int *cx2,
    __constant int *cy2,
    const int num_jumpers_2,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int BC_SPECIFIER,
    const int PSI_SPECIFIER,
    __constant double *parameters)
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
            if (BC_SPECIFIER == 0){ //PERIODIC
                if (temp_x >= nx) temp_x -= nx;
                if (temp_x < 0) temp_x += nx;

                if (temp_y >= ny) temp_y -= ny;
                if (temp_y < 0) temp_y += ny;
            }
            if (BC_SPECIFIER == 1){ //ZERO GRADIENT
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

        //Remember, this is force PER DENSITY to avoid problems
        double force_x_fluid_1 = 0;
        double force_y_fluid_1 = 0;

        double force_x_fluid_2 = 0;
        double force_y_fluid_2 = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        double rho_1_pixel = local_fluid_1[old_2d_buf_index];
        double rho_2_pixel = local_fluid_2[old_2d_buf_index];

        double psi_1_pixel = 0;
        double psi_2_pixel = 0;

        get_psi(PSI_SPECIFIER, rho_1_pixel, rho_2_pixel, &psi_1_pixel, &psi_2_pixel, parameters);

        //Psi at other pixels

        double psi_1 = 0;
        double psi_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers_1; jump_id++){
            int cur_cx = cx1[jump_id];
            int cur_cy = cy1[jump_id];
            double cur_w = pi1[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_rho_1 = local_fluid_1[new_2d_buf_index];
            double cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        for(int jump_id = 0; jump_id < num_jumpers_2; jump_id++){
            int cur_cx = cx2[jump_id];
            int cur_cy = cy2[jump_id];
            double cur_w = pi2[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_rho_1 = local_fluid_1[new_2d_buf_index];
            double cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        force_x_fluid_1 *= -(G_int*psi_1_pixel);
        force_y_fluid_1 *= -(G_int*psi_1_pixel);

        force_x_fluid_2 *= -(G_int*psi_2_pixel);
        force_y_fluid_2 *= -(G_int*psi_2_pixel);

        const int two_d_index = y*nx + x;
        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        if(rho_1_pixel > ZERO_DENSITY){
            Gx_global[three_d_index_fluid_1] += force_x_fluid_1/rho_1_pixel;
            Gy_global[three_d_index_fluid_1] += force_y_fluid_1/rho_1_pixel;
        }
        if(rho_2_pixel > ZERO_DENSITY){
            Gx_global[three_d_index_fluid_2] += force_x_fluid_2/rho_2_pixel;
            Gy_global[three_d_index_fluid_2] += force_y_fluid_2/rho_2_pixel;
        }


    }
}
