#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

__global__ void vadd (int *a, int *b, int *c, int N)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N)
    c[myID] = a[myID] + b[myID];
}

int main (void)
{
  int *ha, *hb, *hc, *da, *db, *dc;     // host (h*) and device (d*) pointers
  int i, N, BLOCK_SIZE = 256;

  printf("Enter the size of the vectors: ");
  scanf("%d", &N);

  ha = (int*)malloc(sizeof(int)*N);
  hb = (int*)malloc(sizeof(int)*N);
  hc = (int*)malloc(sizeof(int)*N);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * N)); 
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &dc, sizeof (int) * N));

  for (i = 0; i < N; i++)
  {
    ha[i] = rand () % 10000;
    hb[i] = rand () % 10000;
  }

  CUDA_CHECK_RETURN (cudaMemcpy (da, ha, sizeof (int) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN (cudaMemcpy (db, hb, sizeof (int) * N, cudaMemcpyHostToDevice));

  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  vadd <<< grid, BLOCK_SIZE >>> (da, db, dc, N);

  CUDA_CHECK_RETURN (cudaDeviceSynchronize ());

  cudaEventRecord(stop);

  CUDA_CHECK_RETURN (cudaGetLastError ());
  CUDA_CHECK_RETURN (cudaMemcpy (hc, dc, sizeof (int) * N, cudaMemcpyDeviceToHost));

  for (i = 0; i < N; i++)
  {
    if (hc[i] != ha[i] + hb[i])
      printf ("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
  }

  CUDA_CHECK_RETURN (cudaFree ((void *) da));
  CUDA_CHECK_RETURN (cudaFree ((void *) db));
  CUDA_CHECK_RETURN (cudaFree ((void *) dc));
  free(ha);
  free(hb);
  free(hc);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time elapsed: %f ms\n", milliseconds);

  CUDA_CHECK_RETURN (cudaDeviceReset ());

  return 0;
}
