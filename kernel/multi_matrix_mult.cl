
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;

__constant const uint MATRIX_COUNT = 4;

__constant const uint REP_NB = 100u;

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

