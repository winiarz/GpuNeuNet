
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;

__constant const uint REP_NB = 100u;

__kernel void matrixMultiply_simple_n_n_n(__global float* g_A,
								          __global float* g_B,
									      __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;

            float r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_A[r_x + MATRIX_SIZE*r_j] * g_B[r_j + MATRIX_SIZE*r_y];
            }

            g_result[r_i] = r_sum;
        }
    }
}

__kernel void matrixMultiply_opt1_n_n_n(__global float* g_A,
								        __global float* g_B,
								        __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i & MATRIX_SIZE_MOD_BIT;
            uint r_y = r_i >> MATRIX_SIZE_LOG;

            float r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_A[r_x + (r_j << MATRIX_SIZE_LOG)] * g_B[r_j + (r_y << MATRIX_SIZE_LOG)];
            }

            g_result[r_i] = r_sum;
        }
    }
}

__kernel void matrixMultiply_simple_s_n_n(__global float* g_A,
								          __global float* g_B,
									      __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;

            float r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_A[r_j + MATRIX_SIZE*r_x] * g_B[r_j + MATRIX_SIZE*r_y];
            }

            g_result[r_i] = r_sum;
        }
    }
}

__kernel void matrixMultiply_opt_s_n_n(__global float* g_A,
								       __global float* g_B,
								       __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i & MATRIX_SIZE_MOD_BIT;
            uint r_y = r_i >> MATRIX_SIZE_LOG;

            float r_sum = 0.0f;

            uint r_Ast = r_x << MATRIX_SIZE_LOG;
            uint r_Bst = r_y << MATRIX_SIZE_LOG;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_A[r_j + r_Ast] * g_B[r_j + r_Bst];
            }

            g_result[r_i] = r_sum;
        }
    }
}

__kernel void matrixMultiply_opt2_s_n_n(__global float* g_A,
								        __global float* g_B,
								        __global float* g_result)
{
    __global float4* g_A4 = (__global float4*) g_A;
    __global float4* g_B4 = (__global float4*) g_B;
    __global float4* g_result4 = (__global float4*) g_result;

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i & MATRIX_SIZE_MOD_BIT;
            uint r_y = r_i >> MATRIX_SIZE_LOG;

            float4 r_sum = 0.0f;

            uint r_Ast = r_x << (MATRIX_SIZE_LOG-2);
            uint r_Bst = r_y << (MATRIX_SIZE_LOG-2);

            for(uint r_j=0; r_j<MATRIX_SIZE/4; r_j++)
            {
                r_sum += g_A4[r_j + r_Ast] * g_B4[r_j + r_Bst];
            }

            g_result[r_i] = r_sum.x + r_sum.y + r_sum.z + r_sum.w;
        }
    }
}

__kernel void matrixMultiply_simple_n_s_n(__global float* g_A,
								          __global float* g_B,
									      __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;

            float r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_A[r_x + MATRIX_SIZE*r_j] * g_B[r_y + MATRIX_SIZE*r_j];
            }

            g_result[r_i] = r_sum;
        }
    }
}

__kernel void matrixMultiply_opt1_n_s_n(__global float* g_A,
								        __global float* g_B,
								        __global float* g_result)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i & MATRIX_SIZE_MOD_BIT;
            uint r_y = r_i >> MATRIX_SIZE_LOG;

            float r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE_SQ; r_j+=MATRIX_SIZE)
            {
                r_sum += g_A[r_x + r_j] * g_B[r_y + r_j];
            }

            g_result[r_i] = r_sum;
        }
    }
}


