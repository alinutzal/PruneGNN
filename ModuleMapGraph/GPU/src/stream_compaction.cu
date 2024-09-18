/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

//-------------------------------------------------------------------------------------------
template <class T, class Tintbool>
__global__ void compact_stream (T *output, T *input, Tintbool *mask, int *mask_sum, int size)
//-------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= size) || (!mask[i])) return;

  output[mask_sum[i]] = input[i];
}

//-------------------------------------------------------------------------------------
template <class T>
__global__ void compact_stream (T *output, T *input, int *mask_sum, int step, int size)
//-------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int shift = i * step;
  if (i >= size) return;

  for (int k=mask_sum[i]; k<mask_sum[i+1]; k++, shift++)
    output[k] = input[shift];
}
