/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

//-----------------------------------------------
template <class T>
__global__ void init_vector (T* vector, int size)
//-----------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    vector[i] = i;
}

//-----------------------------------
__device__ void swap (int &a, int &b)
//-----------------------------------
{
  if (a == b) return;
  int t = a;
  a = b;
  b = t;
}

//----------------------------------------------------------------------------
__device__ void quick_sort (int *M2_SP, int *sorted_M2_SP, int start, int end)
//----------------------------------------------------------------------------
{
  if (start > end) return;

  // select the lefttmost element as pivot
  int pivot = M2_SP[sorted_M2_SP[start]];
  int low = start;
  int high = start;

  // traverse each element of the array
  // compare them with the pivot
  for (int j=start+1; j<end; j++) {
    int SP2 = M2_SP[sorted_M2_SP[j]];
    // if element smaller than pivot is found swap it with the greater element pointed by i
    if (SP2 < pivot) {
      swap (sorted_M2_SP[j], sorted_M2_SP[++high]);
      swap (sorted_M2_SP[high], sorted_M2_SP[low++]);
    }
    else
      if (SP2 == pivot)
        swap (sorted_M2_SP[++high], sorted_M2_SP[j]);
  }

  if (low == start && high == end) return;
  quick_sort (M2_SP, sorted_M2_SP, start, low);
  quick_sort (M2_SP, sorted_M2_SP, high+1, end);
}

//---------------------------------------------------------------------------------------
__global__ void partial_quick_sort (int *sorted_vector, int *vector, int *mask, int size)
//---------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size || mask[i] == mask[i+1]) return;

  quick_sort (vector, sorted_vector, mask[i], mask[i+1]);
}
