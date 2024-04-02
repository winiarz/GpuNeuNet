
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;
__constant const uint REP_NB = 100u;


__kernel void neur_layer_simple (__global float* g_input,
								                 __global float* g_weights,
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
          r_sum += g_input[r_x + MATRIX_SIZE*r_j] * g_weights[r_j + MATRIX_SIZE*r_y];
      }

      float r_res = 1.0f / (1.0f + exp(-0.01f * r_sum ));
      g_result[r_i] = r_res;
  }

  g_result[r_id + MATRIX_SIZE*(MATRIX_SIZE-1)] = 1.0f;
  }
}

__kernel void neur_layer_opt (__global float* g_input,
								                 __global float* g_weights,
									               __global float* g_result)
{
  __local float l_weights[MATRIX_SIZE];
  
  for(uint a=0; a< REP_NB; a++)
  {

  uint r_id = get_global_id(0);
  uint r_size = get_global_size(0);

  for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=r_size)
  {
      uint r_x = r_i % MATRIX_SIZE;
      uint r_y = r_i / MATRIX_SIZE;

      float r_sum = 0.0f;
      
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      l_weights[r_id] = g_weights[r_id + MATRIX_SIZE*r_y];
      work_group_barrier(CLK_LOCAL_MEM_FENCE);


      for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
      {
          r_sum += g_input[r_x + MATRIX_SIZE*r_j] * l_weights[r_j];
      }

      float r_res = 1.0f / (1.0f + exp(-0.01f * r_sum ));
      g_result[r_i] = r_res;
  }

  g_result[r_id + MATRIX_SIZE*(MATRIX_SIZE-1)] = 1.0f;
  }
}

