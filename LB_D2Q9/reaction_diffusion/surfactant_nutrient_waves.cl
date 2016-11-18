__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *rho_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __constant float *w,
           __constant int *cx,
           __constant int *cy,
           const float cs,
           const int nx, const int ny, const int num_populations)
{
    //Input should be a 2d workgroup. But, we loop over a 4d array...
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int two_d_index = y*nx + x;

    if ((x < nx) && (y < ny)){

        const float u = u_global[two_d_index];
        const float v = v_global[two_d_index];

        // rho is three-dimensional now...have to loop over every field.
        for (int field_num=0; field_num < num_populations; field_num++){
            int three_d_index = field_num*nx*ny + two_d_index;
            float rho = rho_global[three_d_index];
            // Now loop over every jumper
            for(int jump_id=0; jump_id < 9; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny + three_d_index;

                float cur_w = w[jump_id];
                int cur_cx = cx[jump_id];
                int cur_cy = cy[jump_id];

                float cur_c_dot_u = cur_cx*u + cur_cy*v;

                float new_feq = cur_w*rho*(1.f + cur_c_dot_u/(cs*cs));

                feq_global[four_d_index] = new_feq;
            }
        }
    }
}


__kernel void
update_hydro(__global float *f_global,
             __global float *u_global,
             __global float *v_global,
             __global float *rho_global,
             const int nx, const int ny, const int num_populations)
{
    // Assumes that u and v are imposed. Can be changed later.
    //This *MUST* be run after move_bc's, as that takes care of BC's
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        // Loop over fields.
        for(int field_num = 0; field_num < num_populations; field_num++){
            int three_d_index = field_num*nx*ny + two_d_index;

            float f_sum = 0;
            for(int jump_id = 0; jump_id < 9; jump_id++){
                f_sum += f_global[jump_id*num_populations*nx*ny + three_d_index];
            }
            rho_global[three_d_index] = f_sum;
        }
    }
}

__kernel void
collide_particles(__global float *f_global,
                  __global __read_only float *feq_global,
                  __global __read_only float *rho_global,
                  const float omega, const float omega_n,
                  const float G,
                  __constant float *w,
                  const int nx, const int ny, const int num_populations)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){

        const int two_d_index = y*nx + x;

        float cur_rho = rho_global[0*ny*nx + two_d_index]; // Density
        float cur_n = rho_global[1*ny*nx + two_d_index]; // Nutrient concentration

        float all_growth = cur_G * cur_rho * cur_n;

        //****** POPULATION ******
        int cur_field = 0
        int three_d_index = cur_field*ny*nx + two_d_index;
        for(int jump_id=0; jump_id < 9; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            float f = f_global[four_d_index];
            float feq = feq_global[four_d_index];
            float cur_w = w[jump_id];

            float relax = f*(1-omega) + omega*feq;
            float growth = relax + cur_w*all_growth;

            f_global[four_d_index] = relax + growth;
        }
        //****** NUTRIENT ******
        int cur_field = 1
        int three_d_index = cur_field*ny*nx + two_d_index;
        for(int jump_id=0; jump_id < 9; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            float f = f_global[four_d_index];
            float feq = feq_global[four_d_index];
            float cur_w = w[jump_id];

            float relax = f*(1-omega_n) + omega_n*feq;
            //Nutrients are depleted at the same rate cells grow
            float growth = relax - cur_w*all_growth;

            f_global[four_d_index] = relax + growth;
        }
    }
}


__kernel void
move(__global __read_only float *f_global,
     __global __write_only float *f_streamed_global,
     __constant int *cx,
     __constant int *cy,
     const int nx, const int ny, const int num_populations)
{
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

            if ((stream_x >= 0) && (stream_x < nx) && (stream_y >= 0) && (stream_y < ny)){ // Stream
                for(int field_num = 0; field_num < num_populations; field_num++){
                    int slice = jump_id*num_populations*nx*ny + field_num*nx*ny;

                    int old_4d_index = slice + y*nx + x;
                    int new_4d_index = slice + stream_y*nx + stream_x;

                    f_streamed_global[new_4d_index] = f_global[old_4d_index];
                }
            }
        }
    }
}

__kernel void
move_bcs(__global float *f_global,
         __constant float *w,
         const int nx, const int ny,
         const int num_populations)
{
    //TODO: Make this efficient. I recognize there are better ways to do this, perhaps a kernel for each boundary...
    //Input should be a 2d workgroup! Everything is done inplace, no need for a second buffer
    //Must be run after move
    //ONLY WORKS IN THE CASE OF NO VELOCITY RIGHT NOW
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Check if on the main surfaces
    const bool on_left = (x==0) && (y >= 1)&&(y < ny-1);
    const bool on_right = (x==nx - 1) && (y >= 1)&&(y < ny -1);
    const bool on_top = (y == ny-1) && (x >= 1) && (x< nx-1);
    const bool on_bottom = (y == 0) && (x >= 1) && (x < nx-1);

    const bool on_main_surface = on_left || on_right || on_top || on_bottom;

    //Check if on the corners
    const bool bottom_left_corner = (x==0) && (y==0);
    const bool bottom_right_corner = (x==nx-1)&&(y==0);
    const bool upper_left_corner = (x==0)&&(y==ny-1);
    const bool upper_right_corner = (x==nx-1)&&(y==ny-1);

    const bool on_corner = bottom_left_corner || bottom_right_corner || upper_left_corner || upper_right_corner;

    const int two_d_index = y*nx + x;
    const int four_d_prefactor = ny*nx*num_populations;

    if (on_main_surface || on_corner){
        for(int field_num = 0; field_num < num_populations; field_num++){

            int three_d_index = field_num*nx*ny + two_d_index;
            // You need to read in all f...except f0 actually
            float f1 = f_global[1*four_d_prefactor + three_d_index];
            float f2 = f_global[2*four_d_prefactor + three_d_index];
            float f3 = f_global[3*four_d_prefactor + three_d_index];
            float f4 = f_global[4*four_d_prefactor + three_d_index];
            float f5 = f_global[5*four_d_prefactor + three_d_index];
            float f6 = f_global[6*four_d_prefactor + three_d_index];
            float f7 = f_global[7*four_d_prefactor + three_d_index];
            float f8 = f_global[8*four_d_prefactor + three_d_index];

            if (on_top){
                f_global[7*four_d_prefactor + three_d_index] = f5;
                f_global[4*four_d_prefactor + three_d_index] = f2;
                f_global[8*four_d_prefactor + three_d_index] = f6;
            }

            //Bottom: no flux
            if (on_bottom){
                f_global[2*four_d_prefactor + three_d_index] = f4;
                f_global[5*four_d_prefactor + three_d_index] = f7;
                f_global[6*four_d_prefactor + three_d_index] = f8;
            }


            //Right: no flux
            if (on_right){
                f_global[3*four_d_prefactor + three_d_index] = f1;
                f_global[6*four_d_prefactor + three_d_index] = f8;
                f_global[7*four_d_prefactor + three_d_index] = f5;
            }

            //Left: no flux
            if (on_left){
                f_global[1*four_d_prefactor + three_d_index] = f3;
                f_global[5*four_d_prefactor + three_d_index] = f7;
                f_global[8*four_d_prefactor + three_d_index] = f6;
            }

            //Corner nodes! Extremely annoying and painful, and likely slow

            // Upper left corner: constant density
            if (upper_left_corner){
                f_global[1*four_d_prefactor + three_d_index] = f3;
                f_global[4*four_d_prefactor + three_d_index] = f2;
                f_global[8*four_d_prefactor + three_d_index] = f6;
            }

            // Upper right corner: constant density
            if (upper_right_corner){
                f_global[3*four_d_prefactor + three_d_index] = f1;
                f_global[4*four_d_prefactor + three_d_index] = f2;
                f_global[7*four_d_prefactor + three_d_index] = f5;
            }

            // Bottom right corner: Constant density
            if (bottom_right_corner){
                f_global[2*four_d_prefactor + three_d_index] = f4;
                f_global[3*four_d_prefactor + three_d_index] = f1;
                f_global[6*four_d_prefactor + three_d_index] = f8;
            }

            //Bottom left corner: Constant density
            if (bottom_left_corner){
                f_global[1*four_d_prefactor + three_d_index] = f3;
                f_global[2*four_d_prefactor + three_d_index] = f4;
                f_global[5*four_d_prefactor + three_d_index] = f7;

            }
        }
    }
}