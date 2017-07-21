#include "cuda_helpers.h"
#include "curand.h"
#include "curand_kernel.h"

#include <thread>
#include <iostream>
#include <mutex>
#include <cassert>
#include "fcc_diffusion.h"
#include "parameters.h"

using namespace std;

const int threads_per_block = 512;
const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

struct GPUData {
    int num_mnps;
    MNP_info* dev_mnp;

    Triple* dev_lattice;
    int* sphereLookup;
    water_info* dev_waters;

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
};

/**
 * Initializes the random state associated with each thread - used so that
 * each thread can generate its own random numbers.
 */
__device__ void initRandomState(int tid, unsigned int seed, curandState_t* states) {
    curand_init(seed, tid, 0, states + tid);
}

/**
 * Returns a double randomly and uniformly distributed from 0 to 1.
 */
__device__ double getUniformDouble(curandState_t* state) {
    return curand_uniform_double(state);
}

/**
 * Returns a double from a standard normal distribution with the given
 * standard deviation.
 *
 * TODO: Check whether this is the correct way to scale a normal distribution
 */
__device__ double getNormalDouble(curandState_t* state, double stdev) {
    return curand_normal_double(state) * stdev;
}

/**
 * Determines whether or not a given water molecule resides within a cell.
 */
__device__ bool in_cell(water_info *w, Triple *lat) {
    int idx = w->nearest;
    double dx = w->x - lat[idx].x;
    double dy = w->y - lat[idx].y;
    double dz = w->z - lat[idx].z;
    return (cell_r * cell_r) > NORMSQ(dx, dy, dz);
}

__device__ water_info rand_displacement(water_info *w, GPUData &d) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    water_info disp;
    double norm;
    if(in_cell(w, d.dev_lattice)) {
        norm = getNormalDouble(d.states + tid, d.in_stdev);
    }
    else {
        norm = getNormalDouble(d.states + tid, d.out_stdev);
    }
    disp.x = getUniformDouble(d.states + tid);
    disp.y = getUniformDouble(d.states + tid);
    disp.z = getUniformDouble(d.states + tid);

    double nConstant = NORMSQ(disp.x, disp.y, disp.z);

    disp.x *= norm / nConstant;
    disp.y *= norm / nConstant;
    disp.z *= norm / nConstant;

    return disp;
}

/**
 * First pass: just have the waters randomly diffusing in the simulation
 * boundary.
 */
__global__ void simulateWaters(GPUData d)  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize random number generator state
    initRandomState(tid, d.seed, d.states);

    // Replace by the number of timesteps to run the simulation for
    for(int i = 0; i < 20 ; i++) {
        
    }
}

int main(void) {
    cout << "Starting GPU Simulation..." << endl;
    FCC lattice(D_cell, D_extra, P_expr);
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

    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);

    // Seed the GPU random number generator
    d.seed = time(NULL) + rd();

    // Allocate waters and copy over to the device
    HANDLE_ERROR(cudaMalloc((void **) &(d.dev_waters),
        num_water * sizeof(water_info)));
    HANDLE_ERROR(cudaMemcpy((void*) waters, (void *) d.dev_waters,
        sizeof(water_info) * num_water,
        cudaMemcpyHostToDevice));

    // Call the kernel and execute the simulation!
    simulateWaters<<<num_blocks, threads_per_block>>>(d);

    // Test copying the waters back to the host
    water_info* testWaters = new water_info[num_water];
    HANDLE_ERROR(cudaMemcpy((void*) testWaters, (void *) d.dev_waters,
        sizeof(water_info) * num_water,
        cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_water; i++) {
        if((waters[i].x != testWaters[i].x)
            || (waters[i].y != testWaters[i].y)
            || (waters[i].z != testWaters[i].z))
            cout << "Memory Error at " << i << endl;
    }

    // Free used resources
    cudaFree(d.dev_waters);
    delete[] linLattice;
    delete[] lookupTable;
    delete[] waters;
    delete mnps;
}
