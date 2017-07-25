#define PI 3.14159

#include "cuda_helpers.h"
#include "curand.h"
#include "curand_kernel.h"

#include <thread>
#include <iostream>
#include <mutex>
#include <cassert>
#include <cmath>
#include "parameters.h"
#include "fcc_diffusion.h"

using namespace std;

/*
nvcc cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -std=c+
+11 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64_x86"
*/

const int threads_per_block = 512;
const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

__global__ void kernel(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if(tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    cout << "Starting GPU Simulation..." << endl;
    //FCC lattice(D_cell, D_extra, P_expr);
    int a[N], res[N];
    for(int i = 0; i < N; i++) {
        a[i] = i;
    }
    int *dev_a;
    int *dev_res;

    HANDLE_ERROR(cudaMalloc((void **) &dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice));

    // Allocate data for result
    HANDLE_ERROR(cudaMalloc((void **) &dev_res, sizeof(int) * N));

    cout << "Kernel prepped" << endl;
    // Run the kernel
    kernel<<<N, 1>>>(dev_a, dev_a, dev_res);

    cout << "Ran kernel!" << endl;

    // Copy result back to the host
    HANDLE_ERROR(cudaMemcpy(res, dev_res, sizeof(int)*N, cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; i++) {
        cout << res[i] << endl;
    }

    cudaFree(dev_a);
    cudaFree(dev_res);
}
