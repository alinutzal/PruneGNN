/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 *
 * prescan algorithm base on CUDA Parallel Prefix Sum (SCAN) - see Mark Harris doc
 ***************************************************************/

#include <stdio.h>
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

#ifdef ZERO_BANK_CONFLICTS
  #define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
  #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif 

//---------------------------------------------------------------------------------
template <class I>
__global__ void prescan_large (int *output, I *input, int n, int size, int *sums)
//---------------------------------------------------------------------------------
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	int partial_sum;
//	if (threadID > size) return;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET (ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET (bi);

	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
	  __syncthreads();
	  if (threadID < d) {
		int ai = offset * (2 * threadID + 1) - 1;
		int bi = offset * (2 * threadID + 2) - 1;
		ai += CONFLICT_FREE_OFFSET (ai);
		bi += CONFLICT_FREE_OFFSET (bi);
		temp[bi] += temp[ai];
	  }
    offset *= 2;
	}

	if (threadID == 0) {
		partial_sum = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET (ai);
			bi += CONFLICT_FREE_OFFSET (bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

// write results to device memory
	if (blockOffset + ai > size) return;

	output[blockOffset + ai] = temp[ai + bankOffsetA];
//	output[blockOffset + bi] = temp[bi + bankOffsetB];

	if (!threadID) {
		__threadfence();
		sums[blockID] = partial_sum;
	}
}

//-------------------------------------------------------------
__global__ void add (int *output, int length, int *n, int size)
//-------------------------------------------------------------
{
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	if (blockOffset + threadID > size) return;
	output[blockOffset + threadID] += n[blockID];
}

//----------------------------------------------------
template <class I>
void scan (int *output, I *input, int blocks, int n)
//----------------------------------------------------
{
  int *sums;
  int nb_sums = (n + blocks - 1) / blocks;
  cudaMalloc (&sums, nb_sums * sizeof(int));

  prescan_large<<<nb_sums, THREADS_PER_BLOCK, 2 * ELEMENTS_PER_BLOCK * sizeof(int)>>> (output, input, THREADS_PER_BLOCK, n, sums);

  int *cumulative_sums;
  int n_sums = n / blocks;

	if (!n_sums) return;
  cudaMalloc (&cumulative_sums, (n_sums+1) * sizeof(int));
	scan (cumulative_sums, sums, blocks, n_sums);
	add<<<n_sums+1,blocks>>> (output, blocks, cumulative_sums, n);

  cudaDeviceSynchronize();
	cudaFree (cumulative_sums);
	cudaFree (sums);
}
