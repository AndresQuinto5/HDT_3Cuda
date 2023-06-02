/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : nvcc hello2.cu -o hello2 -arch=sm_20
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void hello ()
{
  int myID = (blockIdx.z * gridDim.x * gridDim.y +
               blockIdx.y * gridDim.x +
               blockIdx.x) * blockDim.x +
               threadIdx.x;

  if (myID == 99999) {  // ID global máximo para 100,000 hilos (99999 = 100,000 - 1)
    printf("\n");
    printf ("Hello world from thread %i (ID global máximo), my name is Andres Quinto and my student ID is 18288\n", myID);
  }
}

int main ()
{
  dim3 g (313, 1);  // Se necesita una cuadrícula de 313 bloques (313 * 320 = 100,160 hilos)
  dim3 b (320, 1);  // Cada bloque tiene 320 hilos
  hello <<< g, b >>> ();
  cudaDeviceSynchronize();
  return 0;
}
