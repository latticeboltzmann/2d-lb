const float[9] w = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36., 1./36.,1./36.,1./36.]}; // weights for directions
const float cs = 1/np.sqrt(3);
const float cs2 = cs**2;
const float cs22 = 2*cs2;
const float cssq = 2.0/9.0;
const float w0 = 4./9.;
const float w1 = 1./9.;
const float w2 = 1./36.;

const float NUM_JUMPERS = 9;

__kernel void
update_feq(__global __read_write float *feq_global,
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
    float feq = feq_global[x, y, jump_id]

    float ul = u/cs2;
    float vl = v/cs2;
    float uv = ul*vl;
    float usq = u*u;
    float vsq = v*v;
    float sumsq  = (usq+vsq)/cs22;
    float sumsq2 = sumsq*(1.-cs2)/cs2;
    float u2 = usq/cssq;
    float v2 = vsq/cssq;

    if (jump_id == 0){
        feq_global[x, y, jump_id] = w0*rho*(1. - sumsq);
    }
    if (jump_id == 1){
        feq_global[x, y, jump_id] = w1*rho*(1. - sumsq  + u2 + ul)
    }
    if (jump_id == 2){
        feq_global[x, y, jump_id] = w1*rho*(1. - sumsq  + v2 + vl)
    }
    if (jump_id == 3){
        feq_global[x, y, jump_id] = w1*rho*(1. - sumsq  + u2 - ul)
    }
    if (jump_id == 4){

    }

    feq[4, :, :] = w1*rho*(1. - sumsq  + v2 - vl)
    feq[5, :, :] = w2*rho*(1. + sumsq2 + ul + vl + uv)
    feq[6, :, :] = w2*rho*(1. + sumsq2 - ul + vl - uv)
    feq[7, :, :] = w2*rho*(1. + sumsq2 - ul - vl + uv)
    feq[8, :, :] = w2*rho*(1. + sumsq2 + ul - vl - uv)
}