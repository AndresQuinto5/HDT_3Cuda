/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : nvcc hello.cu -o hello -arch=sm_20
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 2048

__global__ void hello()
{
    int myID = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (myID < BLOCK_SIZE) {
        printf("%d\n", myID);
    }
}

int main()
{
    hello<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
    printf("Hola, soy Andres Q!! 18288\n");
    return 0;
}
