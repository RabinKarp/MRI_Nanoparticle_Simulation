#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;

const int num_blocks = 32;
const int threads_per_block = 512;

struct GPUData {
    int num_mnps;
    MNP_info* dev_mnp;

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

    // Free memory resources used over the course of this functions
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

__device__ double field(water_info* loc, MNP_info* mnp) {

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

/**
 * Causes a water molecule to accumulate phase according to the magnetic
 * field that it experiences. Currently implemented to simply sum up
 * the field contributions of all MNPs in the volume.
 */
__device__ void accumulatePhase(water_info* w, int mnp_count, MNP_info* mnps) {
    for(int i = 0; i < mnp_count; i++) {
        double f = field(w, mnps[i]);
    }
}

/**
 * Kernel that actually performs the diffusing water simulation
 */
__global__ void waterSimulate(GPUData d) {
    __shared__ double mags[threads_per_block];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = 0; i < t; i++) {

        if(tid < num_water) {
            for(int i = 0; i < 10; i++) {
              
            }
            // TODO: Take a random step in a direction
            // TODO: Check for cell reflection
            // TODO: Check for MNP reflection
            // TODO: Check for boundary condition crossing
            // TODO: Accumulate phase
        }

        // Sum up the net magnetizations
        sumMag(tid, mags, d.magnetizations)
    }
}

double* gpuSimulate(vector<MNP_info> * mnps, water_info* w) {
    GPUData d = initializeGPU(mnps, w);
    waterSimulate<<<num_blocks, threads_per_block>>>(d);
    return finalizeGPU(d);
}
