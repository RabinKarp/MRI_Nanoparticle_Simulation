#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cunistd>
#include <cstdio>

#include "rand_walk.h"
#include "cuda_helpers.h"

using namespace std;

const int num_blocks = 32;
const int threads_per_block = 512;

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T

struct GPUData {
    int num_mnps;
    MNP_info* dev_mnp;

    // The CUDA random number generator
    curandState_t state;

    double* dev_lattice;
    water_info* dev_waters;
    /**
     * The array of magnetizations is a double array of dimension
     * (t * num_blocks). Each block writes to a unique portion of the shared
     * global memory.
     */
     double* magnetizations;
};

void HANDLE_ERROR(int errorCode) {
    // Now what?
}

/**
 * Prepares data in proper format, loads data onto the GPU and allocates
 * space for the answers that the GPU prepares to compute.
 */
GPUData initializeGPU(vector<MNP_info> *mnpList, water_info* w) {
    GPUData d;
    d.num_mnps = mnpList->size();

    // Initialize the random number generator
    curandCreateGenerator(CURAND_RNG_PSEUDO_XORWOW);

    // Get data into standard C arrays on the host
    MNP_info* mnps = calloc(mnpList->size() , sizeof(MNP_info));
    double* latticePoints = calloc(num_cells * 3, sizeof(double));

    for(int i = 0; i < num_cells * 3; i++) {
        latticePoints[i * 3] = fcc[i][0];
        latticePoints[i * 3 + 1] = fcc[i][1];
        latticePoints[i * 3 + 2] = fcc[i][2];
    }

    for(int i = 0; i < mnpList->size(); i++) {
        mnps[i] = mnpList[i];
    }

    // Allocate the input data on the GPU
    HANDLE_ERROR(cudaMalloc((void **) d.dev_mnp,
        sizeof(MNP_info) * d.num_mnps));
    HANDLE_ERROR(cudaMalloc((void **) d.dev_waters,
        sizeof(water_info) * num_water));
    HANDLE_ERROR(cudaMalloc((void **) d.dev_lattice,
        sizeof(MNP_info) * d.num_mnps));

    // Allocate the output data on the GPU
    HANDLE_ERROR(cudaMalloc((void **) d.magnetizations,
        sizeof(double) * t * num_blocks));

    // Copy waters, MNPs, and cells to device
    HANDLE_ERROR(cudaMemcpy(mnps, d.dev_mnp, sizeof(MNP_info) * d.num_mnps,
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(w, d.dev_waters, sizeof(water_info) * num_water,
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(latticePoints, d.dev_lattice,
        sizeof(double) * num_cells * 3,
        cudaMemcpyHostToDevice));

    // Free memory allocated within this function
    free(latticePoints);
    free(mnps);

    // Return the struct containing device pinters
    return data;
}

/**
 * Computes the array of net magnetizations produced by the GPU and frees
 * all resources allocated on the device.
 */
double* finalizeGPU(GPUData d) {
    double* computedMagnetizations = calloc(t * num_blocks, sizeof(double));
    double* netMagnetizations = calloc(t, sizeof(double));

    // Copy the device memory to the host
    HANDLE_ERROR(cudaMemcpy(computedMagnetizations, d.magnetizations,
        sizeof(double) * t * num_blocks,
        cudaMemcpyDeviceToHost));

    // Compute the array of net magnetizations by summing up the data
    // from all blocks for each time step
    for(int i = 0 ; i < t; i++) {
        for(int j = 0; j < num_blocks; j++) {
            netMagnetizations[i] += computedMagnetizations[i * num_blocks + j];
        }
    }

    // Free device memory
    cudaFree(d.dev_mnp);
    cudaFree(d.dev_lattice);
    cudaFree(d.dev_waters);
    cudaFree(d.magnetizations);

    // Free host memory allocated in this function
    free(computedMagnetizations);

    return netMagnetizations
}

__device__ bool in_cell(water_info *w) {
    double *center = fcc[w->nearest];
    double x = w.x - center[0];
    double y = w.y - center[1];
    double z = w.z - center[2];
    return cell_r * cell_r > NORMSQ(x, y, z);
}

/**
 * Updates the cell closest to a water molecule, which is stored by that
 * molecule for easy reference.
 */
__device__ void updateNearestCell(water_info *w) {

}

/**
 * Returns the random displacement of a water molecule according to a specified
 * normal distribution.
 */
__device__ water_info rand_displacement() {

    // Assume that we somehow already have the displacement norm
    double d;


}

/**
 * Check for cell boundary crossings and apply a reflection appropriately
 *
 * Returns true if a reflection occurs and false otherwise.
 *
 */
__device__ bool cellReflection(water_info* i, water_info* f) {
    // First handle the case where the water diffuses into the cell
    if( ! in_cell(i) && in_cell(f)) {
        // Flip a coin to decide whether or not to diffuse into the cell
    }

    // Handle the case where water diffuses out of the cell
    if(in_cell(i) && ! in_cell(f)) {
        // Flip a coin to decide whether or not to diffuse out of the cell
    }
}

/**
 * Compute reflection off MNPs trivially - that is, by looping over all MNPs
 * and checking the distance norm to them. Note that this function doesn't
 * actually apply the reflection.
 *
 * Returns true if a reflection needs to occur and false otherwise.
 */
__device__ bool mnpReflection(water_info* w, int mnp_count, MNP_info *mnps) {
    bool reflect = false;
    for(int i = 0; i < mnp_count; i++) {
        double dx = mnps[i].x - w->x;
        double dy = mnps[i].y - w->y;
        double dz = mnps[i].z - w->z;
        if(NORMSQ(dx, dy, dz) < pow(cell_r, 2)) {
            reflect = true;
        }
    }
    return reflect;
}

/**
 * Memory reduction: Use the shared memory in each block to sum up the
 * magnetizations for all water molecules processed by the threads in the block.
 * Design pattern from CUDA by Design (c).
 *
 * @param cache A pointer to the shared memory cache for this block
 * @param t     Pointer to global memory to store the target sum
 */
__device__ void sumMag(double* cache, double* t) {
    int tid = threadIdx.x;
    int i = blockDim.x / 2;

    while(i != 0) {
        if(tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }

    // Copy the sum back into global memory
    if(tid == 0) {
        *t = cache[0];
    }
}

__device__ double dipole_field(double dx, double dy, double dz, double M)
{
    double divisor = pow(NORMSQ(dx, dy, dz), 2.5);
    return M * 1e11 * (2*dz*dz - dx*dx - dy*dy) / divisor;
}

/**
 * Causes a water molecule to accumulate phase according to the magnetic
 * field that it experiences. Currently implemented to simply sum up
 * the field contributions of all MNPs in the volume - the trivial
 * implementation.
 */
__device__ void accumulatePhase(water_info* w, int mnp_count, MNP_info* mnps) {
    double B = 0;
    for(int i = 0; i < mnp_count; i++) {
        double dx = mnps[i].x - w->x;
        double dy = mnps[i].y - w->y;
        double dz = mnps[i].z - w->z;
    }
    w->phase += B * 2 * M_PI * g * tau * 1e-3;
}

/**
 * Constrains the water molecules to lie within the bounds of the simulation.
 * Right now, it is implemented to NOT track the number of boundary crossings
 * for each water molecule.
 */
__device__ void boundary_conditions(water_info* w) {
    w->x %= bound;
    w->y %= bound;
    w->z %= bound;
}

/**
 * Kernel that actually performs the diffusing water simulation
 */
__global__ void waterSimulate(GPUData d) {
    __shared__ double mags[threads_per_block];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = 0; i < t; i++) {
        if(tid < num_water) {
            water_info *w = d->waters[tid];
            for(int i = 0; i < t; i++) {
              // TODO: Take a random step in a direction
              // TODO: Check for boundary condition crossing
              boundary_conditions(w);
              // TODO: Check for cell reflection

              // TODO: Check for MNP reflection
              // Accumulate phase
              accumulatePhase(w);

              // TODO: Apply a flip at the Carr-Purcell time

              // Store the magnetizations in a cache
              mags[threadIdx.x] = cos(w->phase);
              __syncthreads();
              // Sum up the net magnetizations, store in the target
              sumMag(tid, mags, d.magnetizations + i * num_blocks + blockIdx.x);
            }
        }
    }
}

double* gpuSimulate(vector<MNP_info> * mnps, water_info* w) {
    GPUData d = initializeGPU(mnps, w);
    waterSimulate<<<num_blocks, threads_per_block>>>(d);
    return finalizeGPU(d);
}
