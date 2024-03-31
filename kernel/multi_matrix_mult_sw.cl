
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;

__constant const uint MATRIX_COUNT = 4;

__constant const uint REP_NB = 25u;

__kernel void multiMatrixMultiply_opt1_cn_s_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    __local float4 l_mA4[MATRIX_SIZE];
    __local float4 l_mA4_2[MATRIX_SIZE];

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=(2*r_size))
        {
            uint r_x = r_i >> MATRIX_SIZE_LOG;
            uint r_y = r_i & MATRIX_SIZE_MOD_BIT;

            uint r_x2 = (r_i+r_size) >> MATRIX_SIZE_LOG;
            uint r_y2 = (r_i+r_size) & MATRIX_SIZE_MOD_BIT;

            float4 r_sum = 0.0f;
            float4 r_sum2 = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);
            l_mA4  [r_id] = g_mA4[r_x  + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_2[r_id] = g_mA4[r_x2 + (r_id<<MATRIX_SIZE_LOG)];
            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
              float r_b_yj = g_B[r_y  + MATRIX_SIZE*r_j ];

              r_sum  += l_mA4  [r_j] * r_b_yj;//g_B[r_y  + MATRIX_SIZE*r_j ];
              r_sum2 += l_mA4_2[r_j] * r_b_yj;//g_B[r_y2 + MATRIX_SIZE*r_j ];
            }

            g_mResult[r_x  + (r_y <<MATRIX_SIZE_LOG)] = r_sum;
            g_mResult[r_x2 + (r_y2<<MATRIX_SIZE_LOG)] = r_sum2;

        }
    }
}

__kernel void multiMatrixMultiply_opt2_cn_s_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    __local float4 l_mA4[MATRIX_SIZE];
    __local float4 l_mA4_2[MATRIX_SIZE];
    __local float4 l_mA4_3[MATRIX_SIZE];
    __local float4 l_mA4_4[MATRIX_SIZE];

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=(4*r_size))
        {
            uint r_x = r_i >> MATRIX_SIZE_LOG;
            uint r_y = r_i & MATRIX_SIZE_MOD_BIT;

            uint r_x2 = (r_i+r_size) >> MATRIX_SIZE_LOG;
            uint r_x3 = (r_i+2*r_size) >> MATRIX_SIZE_LOG;
            uint r_x4 = (r_i+3*r_size) >> MATRIX_SIZE_LOG;

            float4 r_sum = 0.0f;
            float4 r_sum2 = 0.0f;
            float4 r_sum3 = 0.0f;
            float4 r_sum4 = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);
            l_mA4  [r_id] = g_mA4[r_x  + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_2[r_id] = g_mA4[r_x2 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_3[r_id] = g_mA4[r_x3 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_4[r_id] = g_mA4[r_x4 + (r_id<<MATRIX_SIZE_LOG)];
            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
              float r_b_yj = g_B[r_y  + MATRIX_SIZE*r_j ];

              r_sum  += l_mA4  [r_j] * r_b_yj; //g_B[r_j + MATRIX_SIZE*r_y ];
              r_sum2 += l_mA4_2[r_j] * r_b_yj; //g_B[r_j + MATRIX_SIZE*r_y2];
              r_sum3 += l_mA4_3[r_j] * r_b_yj;
              r_sum4 += l_mA4_4[r_j] * r_b_yj;
            }

            g_mResult[r_x  + (r_y<<MATRIX_SIZE_LOG)] = r_sum;
            g_mResult[r_x2 + (r_y<<MATRIX_SIZE_LOG)] = r_sum2;
            g_mResult[r_x3 + (r_y<<MATRIX_SIZE_LOG)] = r_sum3;
            g_mResult[r_x4 + (r_y<<MATRIX_SIZE_LOG)] = r_sum4;

        }
    }
}

__kernel void multiMatrixMultiply_opt3_cn_s_cn(__global float4* g_mA4,
                                               __global float* g_B,
                                               __global float4* g_mResult)
{
    __local float4 l_mA4[MATRIX_SIZE];
    __local float4 l_mA4_2[MATRIX_SIZE];
    __local float4 l_mA4_3[MATRIX_SIZE];
    __local float4 l_mA4_4[MATRIX_SIZE];
    __local float4 l_mA4_5[MATRIX_SIZE];
    __local float4 l_mA4_6[MATRIX_SIZE];
    __local float4 l_mA4_7[MATRIX_SIZE];
    __local float4 l_mA4_8[MATRIX_SIZE];

    for(uint a=0; a< REP_NB; a++)
    {
        uint r_id = get_global_id(0);
        uint r_size = get_global_size(0);

        for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=(8*r_size))
        {
            uint r_x = r_i >> MATRIX_SIZE_LOG;
            uint r_y = r_i & MATRIX_SIZE_MOD_BIT;

            uint r_x2 = (r_i+r_size) >> MATRIX_SIZE_LOG;
            uint r_x3 = (r_i+2*r_size) >> MATRIX_SIZE_LOG;
            uint r_x4 = (r_i+3*r_size) >> MATRIX_SIZE_LOG;
            uint r_x5 = (r_i+4*r_size) >> MATRIX_SIZE_LOG;
            uint r_x6 = (r_i+5*r_size) >> MATRIX_SIZE_LOG;
            uint r_x7 = (r_i+6*r_size) >> MATRIX_SIZE_LOG;
            uint r_x8 = (r_i+7*r_size) >> MATRIX_SIZE_LOG;

            float4 r_sum = 0.0f;
            float4 r_sum2 = 0.0f;
            float4 r_sum3 = 0.0f;
            float4 r_sum4 = 0.0f;
            float4 r_sum5 = 0.0f;
            float4 r_sum6 = 0.0f;
            float4 r_sum7 = 0.0f;
            float4 r_sum8 = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);
            l_mA4  [r_id] = g_mA4[r_x  + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_2[r_id] = g_mA4[r_x2 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_3[r_id] = g_mA4[r_x3 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_4[r_id] = g_mA4[r_x4 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_5[r_id] = g_mA4[r_x5 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_6[r_id] = g_mA4[r_x6 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_7[r_id] = g_mA4[r_x7 + (r_id<<MATRIX_SIZE_LOG)];
            l_mA4_8[r_id] = g_mA4[r_x8 + (r_id<<MATRIX_SIZE_LOG)];
            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
            {
              float r_b_yj = g_B[r_y  + MATRIX_SIZE*r_j ];

              r_sum  += l_mA4  [r_j] * r_b_yj;
              r_sum2 += l_mA4_2[r_j] * r_b_yj;
              r_sum3 += l_mA4_3[r_j] * r_b_yj;
              r_sum4 += l_mA4_4[r_j] * r_b_yj;
              r_sum5 += l_mA4_5[r_j] * r_b_yj;
              r_sum6 += l_mA4_6[r_j] * r_b_yj;
              r_sum7 += l_mA4_7[r_j] * r_b_yj;
              r_sum8 += l_mA4_8[r_j] * r_b_yj;
            }

            g_mResult[r_x  + (r_y<<MATRIX_SIZE_LOG)] = r_sum;
            g_mResult[r_x2 + (r_y<<MATRIX_SIZE_LOG)] = r_sum2;
            g_mResult[r_x3 + (r_y<<MATRIX_SIZE_LOG)] = r_sum3;
            g_mResult[r_x4 + (r_y<<MATRIX_SIZE_LOG)] = r_sum4;
            g_mResult[r_x5 + (r_y<<MATRIX_SIZE_LOG)] = r_sum5;
            g_mResult[r_x6 + (r_y<<MATRIX_SIZE_LOG)] = r_sum6;
            g_mResult[r_x7 + (r_y<<MATRIX_SIZE_LOG)] = r_sum7;
            g_mResult[r_x8 + (r_y<<MATRIX_SIZE_LOG)] = r_sum8;
        }
    }

}

