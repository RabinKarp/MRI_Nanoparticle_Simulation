
/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "curand.h"
#include "cuda_helpers.h"

using namespace std;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
    size_t n = 100000000;
    size_t i;
    curandGenerator_t gen;
    double *devData, *hostData;

    /* Allocate n floats on host */
    hostData = (double *) calloc(n, sizeof(double));
    cudaEvent_t start, stop;

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));

    /* Create pseudo-random number generator */
    /*CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));*/
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_XORWOW));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                time(NULL)));

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    /* invoke the kernel to get some random numbers */
    HANDLE_ERROR(cudaEventRecord(start, 0));

    cout << "Generating random numbers!" << endl;
    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    cout << "RNG complete! Elapsed time: "
      << elapsedTime << " ms" << endl;

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double),
        cudaMemcpyDeviceToHost));

    /* Show result */
    for(i = 0; i < 200; i++) {
        printf("%1.8f ", hostData[i]);
    }
    printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    return EXIT_SUCCESS;
}
