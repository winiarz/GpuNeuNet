
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;

__constant const uint MATRIX_COUNT = 4;

__constant const uint REP_NB = 25u;

__kernel void multiMatrixMultiply_simple_n_n_n(__global float* g_mA,
                                               __global float* g_B,
                                               __global float* g_mResult)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_k=0; r_k<MATRIX_COUNT; r_k++)
        {
          for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
          {
              uint r_x = r_i % MATRIX_SIZE;
              uint r_y = r_i / MATRIX_SIZE;

              float r_sum = 0.0f;

              for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
              {
                  r_sum += g_mA[r_x + MATRIX_SIZE*r_j + MATRIX_SIZE_SQ*r_k] * g_B[r_j + MATRIX_SIZE*r_y];
              }

              g_mResult[r_i+MATRIX_SIZE_SQ*r_k] = r_sum;
          }
      }
    }
}

__kernel void multiMatrixMultiply_opt1_n_n_n(__global float* g_mA,
                                             __global float* g_B,
                                             __global float* g_mResult)
{
    __local float l_B[MATRIX_SIZE];

   for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_k=0; r_k<MATRIX_COUNT; r_k++)
        {
          for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
          {
              uint r_x = r_i % MATRIX_SIZE;
              uint r_y = r_i / MATRIX_SIZE;

              float r_sum = 0.0f;
              
              barrier(CLK_LOCAL_MEM_FENCE);
              l_B[r_id] = g_B[r_id + (r_y * MATRIX_SIZE)];
              barrier(CLK_LOCAL_MEM_FENCE);

              for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
              {
                  r_sum += g_mA[r_x + MATRIX_SIZE*r_j + MATRIX_SIZE_SQ*r_k] * l_B[r_j];
              }

              g_mResult[r_i+MATRIX_SIZE_SQ*r_k] = r_sum;
          }
      }
    }
}

__kernel void multiMatrixMultiply_simple_cn_n_cn(__global float* g_mA,
                                                 __global float* g_B,
                                                 __global float* g_mResult)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_k=0; r_k<MATRIX_COUNT; r_k++)
        {
          for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
          {
              uint r_x = r_i % MATRIX_SIZE;
              uint r_y = r_i / MATRIX_SIZE;

              float r_sum = 0.0f;

              for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
              {
                  r_sum += g_mA[r_k + r_x*MATRIX_COUNT + r_j*MATRIX_COUNT*MATRIX_SIZE] * g_B[r_j + MATRIX_SIZE*r_y];
              }

              g_mResult[r_i*MATRIX_COUNT +r_k] = r_sum;
          }
      }
    }
}

__kernel void multiMatrixMultiply_opt1_cn_n_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;

            float4 r_sum = 0.0f;

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_mA4[r_x + r_j*MATRIX_SIZE] * g_B[r_j + MATRIX_SIZE*r_y];
            }


            g_mResult[r_i] = r_sum;
        }
    }
}

__kernel void multiMatrixMultiply_opt2_cn_n_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    __local float l_B[MATRIX_SIZE];

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;

            float4 r_sum = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);
            l_B[r_id] = g_B[r_id + (r_y * MATRIX_SIZE)];
            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum += g_mA4[r_x + r_j*MATRIX_SIZE] * l_B[r_j];
            }

            g_mResult[r_i] = r_sum;
        }
    }
}

__kernel void multiMatrixMultiply_opt3_cn_n_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    __local float l_B[MATRIX_SIZE];
    __local float l_B2[MATRIX_SIZE];

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size*2)
        {
            uint r_x = r_i % MATRIX_SIZE;
            uint r_x2 = (r_i+r_size) % MATRIX_SIZE;
            uint r_y = r_i / MATRIX_SIZE;
            uint r_y2 = (r_i+r_size) / MATRIX_SIZE;

            float4 r_sum = 0.0f;
            float4 r_sum2 = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);
            l_B [r_id] = g_B[r_id  + (r_y  * MATRIX_SIZE)];
            l_B2[r_id] = g_B[r_id  + (r_y2 * MATRIX_SIZE)];
            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum  += g_mA4[r_x  + r_j*MATRIX_SIZE] * l_B[r_j];
            }
            g_mResult[r_i] = r_sum;


            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
                r_sum2 += g_mA4[r_x2 + r_j*MATRIX_SIZE] * l_B2[r_j];
            }

            g_mResult[r_i+r_size] = r_sum2;
        }
    }
}

