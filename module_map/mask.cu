/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

//--------------------------------------------------------
template <class T>
__global__ void reverse_mask (T* umask, T* mask, int size)
//--------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    umask[i] = !mask[i];
}

//------------------------------------------------------------------------
template <class T>
__global__ void merge_masks (T* merged_mask, T* mask1, T* mask2, int size)
//------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    merged_mask[i] = mask1[i] || mask2[i];
}

//-------------------------------------------------------
template <class T>
__global__ void tag_mask (T* mask, int* vector, int size)
//-------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    mask[vector[i]] = true;  
}
