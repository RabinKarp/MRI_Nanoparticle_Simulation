#define PI 3.14159
#define HIGH 5000000

#include "cuda_helpers.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <time.h>
#include <dos.h>
#include <windows.h>

#include <stdlib.h>
#include <fstream>
#include <thread>
#include <iostream>
#include <mutex>
#include <cassert>
#include <cmath>
#include "parameters.h"
#include "fcc_diffusion.h"
#include "gpu_random.h"

using namespace std;

/*
nvcc cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64_x86"
*/

const int threads_per_block = 32;
const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

// Each kernel execution handles AT MOST this many timesteps
const int sprintSteps = 10000;
const int num_uniform_doubles = 4; // # of uniform doubles per water per tstep
const int num_normal_doubles = 1;  // # of normal  doubles per water per tstep

__constant__ Triple dev_lattice[num_cells];

struct GPUData {
    int nBlocks;

    int num_mnps;
    MNP_info* dev_mnp;

    int num_cells;
    double cell_r;
    double bound;
    int* sphereLookup;

    int num_waters;
    water_info* waters;

    double in_stdev;
    double out_stdev;

    unsigned int seed;
    curandState_t* states;

    /**
     * The array of magnetizations is a double array of dimension
     * (t * num_blocks). Each block writes to a unique portion of the shared
     * global memory.
     */
     double* magnetizations;
     long long int timesteps;

     double* uniform_doubles;
     double* normal_doubles;

     // Memory for debugging purposes only
     int *flags;
};

void setParameters(GPUData &d) {
    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);
    d.num_cells = num_cells;
    d.num_waters = num_water;
    d.timesteps = sprintSteps;
    d.cell_r = cell_r;
    d.bound = bound;
    d.nBlocks = num_blocks;
}

/**
 * Trivial implementation of nearest cell finder
 */
__device__ void updateNearest(water_info &w, GPUData &d) {
    int cIndex = 0;
    double cDist = HIGH;
    for(int i = 0; i < 172; i++) {
        double dx = w.x - dev_lattice[i].x;
        double dy = w.y - dev_lattice[i].y;
        double dz = w.z - dev_lattice[i].z;
        if(NORMSQ(dx, dy, dz) < cDist) {
            cDist = NORMSQ(dx, dy, dz);
            cIndex = i;
        }
    }
    w.nearest = cIndex;
}

__device__ bool in_cell(water_info *w, GPUData &d) {
    int idx = w->nearest;
    double dx = w->x - dev_lattice[idx].x;
    double dy = w->y - dev_lattice[idx].y;
    double dz = w->z - dev_lattice[idx].z;
    return (d.cell_r * d.cell_r) > NORMSQ(dx, dy, dz);
}

__device__ water_info rand_displacement(water_info *w, curandState_t *state, GPUData &d) {
    water_info disp;
    disp.x = 0;
    disp.y = 0;
    disp.z = 0;

    double norm;

    /*double x = curand_uniform_double(state);
    double y = curand_uniform_double(state);
    double z = curand_uniform_double(state);


    if(in_cell(w, d)) {
        norm = getNormalDouble(state, d.in_stdev);
    }
    else {
        norm = getNormalDouble(state, d.out_stdev);
    }*/

    /*
    disp.x = getUniformDouble(d.states + tid);
    disp.y = getUniformDouble(d.states + tid);
    disp.z = getUniformDouble(d.states + tid);

    double nConstant = NORMSQ(disp.x, disp.y, disp.z);

    disp.x *= norm / nConstant;
    disp.y *= norm / nConstant;
    disp.z *= norm / nConstant;*/

    return disp;
}

__device__ void boundary_conditions(water_info &w, GPUData &d) {
    w.x = w.x - d.bound * ((int) w.x / d.bound);
    w.y = w.y - d.bound * ((int) w.y / d.bound);
    w.z = w.z - d.bound * ((int) w.z / d.bound);
}

__global__ void simulateWaters(GPUData d)  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < d.num_waters) {
        d.flags[tid] = -1;
        water_info &w = d.waters[tid];
        curandState_t *state = &(d.states[tid]);

        // Initialize random number generator state
        updateNearest(w, d);
        int x = 0;

        for(long long int i = 0; i < d.timesteps; i++) {
            x++;
            water_info disp = rand_displacement(&w, state, d);
            updateNearest(w, d);
            //w.x += disp.x;
            //w.y += disp.y;
            //w.z += disp.z;
            //boundary_conditions(w, d);

        }
        d.flags[tid] = x;
    }
}

void finalizeGPU(GPUData &d) {
    cudaFree(d.states);
    cudaFree(d.waters);
    cudaFree(d.flags);
    cudaFree(d.uniform_doubles);
}

int main(void) {
    cout << "Starting GPU Simulation..." << endl;
    FCC lattice(D_cell, D_extra, P_expr);
    cudaEvent_t start, stop;

    // Initialize PRNG seed for MNPs and waters
    std::random_device rd;
    XORShift<uint64_t> gen(time(NULL) + rd());

    // The simulation has 3 distinct components: the lattice, the water
    // molecules, and the nanoparticles
    vector<MNP_info> *mnps = lattice.init_mnps(gen);
    water_info *waters = lattice.init_molecules(bound, num_water, mnps, gen);
    Triple* linLattice = lattice.linearLattice();
    int* lookupTable = lattice.linearLookupTable();

    GPUData d;
    setParameters(d);
    // Seed the GPU random number generator
    d.seed = time(NULL) + rd();

    int totalThreads = num_blocks * threads_per_block;

    int totalUniform =  num_uniform_doubles * num_water * ;
    // Allocations: Perform all allocations here
    HANDLE_ERROR(cudaMalloc((void **) &(d.waters),
        num_water * sizeof(water_info)));
    HANDLE_ERROR(cudaMalloc((void **) &(dev_lattice),
        num_cells * sizeof(Triple)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.flags),
        num_water * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &d.uniform_doubles, n*sizeof(double)));

    // Initialize performance timers
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Perform all memory copies here
    HANDLE_ERROR(cudaMemcpy(d.waters, waters,
        sizeof(water_info) * num_water,
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_lattice, linLattice,
        sizeof(Triple) * num_cells));

    int flags[num_water];

    cout << "Kernel prepped!" << endl;
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Run the kernel in sprints due to memory limits and timeout issues
    for(int i = 0; i < 1; i++) {
        cout << "Starting sprint " << i << "." << endl;
        getUniformDoubles(sprintSteps, d.uniform_doubles);
        simulateWaters<<<num_blocks, threads_per_block>>>(d);
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    cout << "Kernel execution complete! Elapsed time: "
        << elapsedTime << " ms" << endl;

    // Copy back the array of flags to catch any errors
    HANDLE_ERROR(cudaMemcpy(flags, d.flags,
        sizeof(int) * num_water,
        cudaMemcpyDeviceToHost));

    bool success = true;
    for(int i = 0; i < num_water; i++) {
        if(flags[i] != t)
            success = false;
    }
    cout << "Success State: " << success << endl;

    finalizeGPU(d);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    delete[] linLattice;
    delete[] lookupTable;
    delete[] waters;
    delete mnps;
}
