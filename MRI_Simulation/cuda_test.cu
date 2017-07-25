#define PI 3.14159
#define HIGH 5000000

#include "cuda_helpers.h"

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

/**
 *  nvcc cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -arch=sm_61 -lcurand -ccbin "
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64_x86"
 */

#define threads_per_block 256
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

    double reflectIO;
    double reflectOI;

    int num_waters;
    water_info* waters;

    double in_stdev;
    double out_stdev;

    unsigned int seed;

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

    d.reflectIO = 1 - sqrt(tau / (6*D_in)) * 4 * P_expr;
    d.reflectOI = 1 - ((1 - reflectIO) * sqrt(D_in/D_out));

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
__device__ void updateNearest(water_info *w, GPUData &d) {
    int cIndex = 0;
    double cDist = HIGH;
    w->in_cell = false;
    for(int i = 0; i < 172; i++) {
        double dx = w->x - dev_lattice[i].x;
        double dy = w->y - dev_lattice[i].y;
        double dz = w->z - dev_lattice[i].z;
        if(NORMSQ(dx, dy, dz) < cDist) {
            cDist = NORMSQ(dx, dy, dz);
            cIndex = i;
        }
    }
    if(cDist < d.cell_r * d.cell_r)
        w->in_cell = true;
    w->nearest = cIndex;
}

__device__ bool cell_reflect(water_info *i, water_info *f) {
    double coin = d.uniform_doubles[tStep * d.num_waters * 4 + tid * 4 + 3];
    bool flip = (i->in_cell && (! f->in_cell) && coin < d.reflectIO)
                    || ((! i->in_cell) && f->in_cell && coin < d.reflectOI));
    return flip;
}

__device__ bool mnp_reflect(water_info *w, int num_mnps, MNP_info *mnps) {
    return false; // TODO: Fix this!!!
}

__device__ water_info rand_displacement(int tid, int tStep, water_info *w, GPUData &d) {
    water_info disp;
    double norm = abs(d.normal_doubles[tStep * d.num_waters + tid]);
    int baseU = tStep * d.num_waters * 4 + tid * 4;

    disp.x = d.uniform_doubles[baseU];
    disp.y = d.uniform_doubles[baseU + 1];
    disp.z = d.uniform_doubles[baseU + 2];

    if(in_cell(w, d)) {
        norm *= d.in_stdev;
    }
    else {
        norm *= d.out_stdev;
    }

    double nConstant = norm / NORMSQ(disp.x, disp.y, disp.z);

    disp.x *= nConstant;
    disp.y *= nConstant;
    disp.z *= nConstant;

    return disp;
}

__device__ void boundary_conditions(water_info *w, GPUData &d) {
    w->x = fmod(w->x, d.bound);
    w->y = fmod(w->y, d.bound);
    w->z = fmod(w->z, d.bound);
}

__device__ void sumMagnetizations(double *input, int tStep, double *target) {

}

__global__ void simulateWaters(GPUData d)  {
    __shared__ double mags[threads_per_block];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    water_info w;
    int x = 0;

    if(tid < d.num_waters) {
        // Copy water to chip memory
        w = &(d.waters[tid]);
        d.flags[tid] = -1;
        updateNearest(&w, d);
    }

    for(int i = 0; i < d.timesteps; i++) {
        if(tid < d.num_waters) {
            x++;
            water_info init = w;
            water_info disp = rand_displacement(tid, i, &w, d);
            w.x += disp.x;
            w.y += disp.y;
            w.z += disp.z;
            boundary_conditions(&w, d);
            updateNearest(&w, d);

            if(cell_reflect(&init, &w) || mnp_reflect(&w, ))

            __syncthreads();

            // Copy the magnetization to shared memory
            mags[i] = cos(w.phase);

            // Perform a memory reduction
            sumMagnetizations(mags, i, d.magnetizations);
        }
    }

    if(tid < d.num_waters) {
        d.flags[tid] = x;
        d.waters[tid] = *w;
    }
}


void finalizeGPU(GPUData &d) {
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

    int totalUniform =  num_uniform_doubles * num_water * sprintSteps;
    int totalNormal = num_normal_doubles * num_water * sprintSteps;
    // Allocations: Perform all allocations here
    HANDLE_ERROR(cudaMalloc((void **) &(d.waters),
        num_water * sizeof(water_info)));
    HANDLE_ERROR(cudaMalloc((void **) &(dev_lattice),
        num_cells * sizeof(Triple)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.flags),
        num_water * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &d.uniform_doubles, totalUniform*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &d.normal_doubles, totalNormal*sizeof(double)));

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

    // Run the kernel in sprints due to memory limits and timeout issues
    for(int i = 0; i < 1; i++) {
        cout << "Starting sprint " << (i+1) << "." << endl;
        getUniformDoubles(totalUniform, d.uniform_doubles);
        getNormalDoubles(totalNormal, d.normal_doubles);

        HANDLE_ERROR(cudaEventRecord(start, 0));
        simulateWaters<<<num_blocks, threads_per_block>>>(d);

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
        cout << "Success State: " << success << endl << "===========" << endl;
    }

    finalizeGPU(d);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    delete[] linLattice;
    delete[] lookupTable;
    delete[] waters;
    delete mnps;
}
