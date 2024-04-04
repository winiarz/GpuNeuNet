
__constant const uint MATRIX_SIZE = 256u;
__constant const uint MATRIX_SIZE_SQ = 256u*256u;
__constant const uint MATRIX_SIZE_LOG = 8u;
__constant const uint MATRIX_SIZE_MOD_BIT = 0x000000ff;
__constant const uint REP_NB = 100u;

__kernel void neuron_network_simple (__global float* g_input,
								                     __global float* g_weights,
									                   __global float* g_result)
{
}

