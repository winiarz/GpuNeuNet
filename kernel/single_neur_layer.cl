
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

__kernel void neur_layer_opt2 (__global float* g_input,
								               __global float* g_weights,
									             __global float* g_result)
{
  __local float l_weights[MATRIX_SIZE<<1];
  
  for(uint a=0; a< REP_NB; a++)
  {

    uint r_id = get_global_id(0);
    uint r_size = get_global_size(0);

    for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=2*r_size)
    {
        uint r_x = r_i % MATRIX_SIZE;
        uint r_y = r_i / MATRIX_SIZE;
        uint r_y2 = r_y+1;

        float2 r_sum = 0.0f;
        
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        l_weights[r_id] = g_weights[r_id + MATRIX_SIZE*r_y];
        l_weights[r_id+MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y2];
        work_group_barrier(CLK_LOCAL_MEM_FENCE);


        for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
        {
            float r_inp = g_input[r_x + MATRIX_SIZE*r_j];

            r_sum.x += r_inp * l_weights[r_j];
            r_sum.y += r_inp * l_weights[r_j+MATRIX_SIZE];
        }

        float2 r_res = 1.0f / (1.0f + exp(-0.01f * r_sum ));

        g_result[r_i] = r_res.x;
        g_result[r_i+MATRIX_SIZE] = r_res.y;
    }

    g_result[r_id + MATRIX_SIZE*(MATRIX_SIZE-1)] = 1.0f;
  }
}

__kernel void neur_layer_opt3 (__global float* g_input,
								               __global float* g_weights,
									             __global float* g_result)
{
  __local float l_weights[MATRIX_SIZE<<2];
  
  for(uint a=0; a< REP_NB; a++)
  {

    uint r_id = get_global_id(0);
    uint r_size = get_global_size(0);

    for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=(r_size<<2))
    {
        uint r_x = r_i % MATRIX_SIZE;
        uint r_y = r_i / MATRIX_SIZE;
        uint r_y2 = r_y+1;
        uint r_y3 = r_y+2;
        uint r_y4 = r_y+3;

        float4 r_sum = 0.0f;
        
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        l_weights[r_id] = g_weights[r_id + MATRIX_SIZE*r_y];
        l_weights[r_id+MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y2];
        l_weights[r_id+2*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y3];
        l_weights[r_id+3*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y4];
        work_group_barrier(CLK_LOCAL_MEM_FENCE);


        for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
        {
            float r_inp = g_input[r_x + MATRIX_SIZE*r_j];
            r_sum.x += r_inp * l_weights[r_j];
            r_sum.y += r_inp * l_weights[r_j+MATRIX_SIZE];
            r_sum.z += r_inp * l_weights[r_j+2*MATRIX_SIZE];
            r_sum.w += r_inp * l_weights[r_j+3*MATRIX_SIZE];
        }

        float4 r_res = 1.0f / (1.0f + exp(-0.01f * r_sum ));

        g_result[r_i] = r_res.x;
        g_result[r_i+MATRIX_SIZE] = r_res.y;
        g_result[r_i+2*MATRIX_SIZE] = r_res.z;
        g_result[r_i+3*MATRIX_SIZE] = r_res.w;
    }

    g_result[r_id + MATRIX_SIZE*(MATRIX_SIZE-1)] = 1.0f;
  }
}

__kernel void neur_layer_opt4 (__global float* g_input,
								               __global float* g_weights,
									             __global float* g_result)
{
  __local float l_weights[MATRIX_SIZE<<3];
  
  for(uint a=0; a< REP_NB; a++)
  {

    uint r_id = get_global_id(0);
    uint r_size = get_global_size(0);

    for(uint r_i=r_id; r_i<MATRIX_SIZE_SQ; r_i+=(r_size<<3))
    {
        uint r_x = r_i % MATRIX_SIZE;
        uint r_y = r_i / MATRIX_SIZE;
        uint r_y2 = r_y+1;
        uint r_y3 = r_y+2;
        uint r_y4 = r_y+3;
        uint r_y5 = r_y+4;
        uint r_y6 = r_y+5;
        uint r_y7 = r_y+6;
        uint r_y8 = r_y+7;

        float4 r_sum = 0.0f;
        float4 r_sum2 = 0.0f;
        
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        l_weights[r_id] = g_weights[r_id + MATRIX_SIZE*r_y];
        l_weights[r_id+MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y2];
        l_weights[r_id+2*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y3];
        l_weights[r_id+3*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y4];
        l_weights[r_id+4*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y5];
        l_weights[r_id+5*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y6];
        l_weights[r_id+6*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y7];
        l_weights[r_id+7*MATRIX_SIZE] = g_weights[r_id + MATRIX_SIZE*r_y8];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);


        for(uint r_j=0; r_j<MATRIX_SIZE; r_j++)
        {
            float r_inp = g_input[r_x + MATRIX_SIZE*r_j];
            r_sum.x += r_inp * l_weights[r_j];
            r_sum.y += r_inp * l_weights[r_j+MATRIX_SIZE];
            r_sum.z += r_inp * l_weights[r_j+2*MATRIX_SIZE];
            r_sum.w += r_inp * l_weights[r_j+3*MATRIX_SIZE];
            r_sum2.x += r_inp * l_weights[r_j+4*MATRIX_SIZE];
            r_sum2.y += r_inp * l_weights[r_j+5*MATRIX_SIZE];
            r_sum2.z += r_inp * l_weights[r_j+6*MATRIX_SIZE];
            r_sum2.w += r_inp * l_weights[r_j+7*MATRIX_SIZE];
        }

        float4 r_res  = 1.0f / (1.0f + exp(-0.01f * r_sum ));
        float4 r_res2 = 1.0f / (1.0f + exp(-0.01f * r_sum2 ));

        g_result[r_i] = r_res.x;
        g_result[r_i+MATRIX_SIZE] = r_res.y;
        g_result[r_i+2*MATRIX_SIZE] = r_res.z;
        g_result[r_i+3*MATRIX_SIZE] = r_res.w;
        g_result[r_i+4*MATRIX_SIZE] = r_res2.x;
        g_result[r_i+5*MATRIX_SIZE] = r_res2.y;
        g_result[r_i+6*MATRIX_SIZE] = r_res2.z;
        g_result[r_i+7*MATRIX_SIZE] = r_res2.w;
    }

    g_result[r_id + MATRIX_SIZE*(MATRIX_SIZE-1)] = 1.0f;
  }
}

