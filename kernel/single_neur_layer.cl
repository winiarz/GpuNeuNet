__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;


__kernel void neur_layer_simple (__global float* g_input,
								                 __global float* g_weights,
									               __global float* g_result)
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

