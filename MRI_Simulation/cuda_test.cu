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

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T

struct GPUData {
    int num_mnps;
    MNP_info* dev_mnp;

    int num_cells;
    Triple* dev_lattice;
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
     int timesteps;
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
    if(tid < d.num_waters) {
        water_info &w = d.waters[tid];
        // Initialize random number generator state
        initRandomState(tid, d.seed, d.states);
        // Replace by the number of timesteps to run the simulation for
        for(int i = 0; i < d.timesteps; i++) {
            water_info disp = rand_displacement(&w, d);
            // TODO: Add the displacement
        }
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
__device__ void accumulatePhase(water_info* w, MNP_info* mnps, int mnp_count) {
    double B = 0;
    for(int i = 0; i < mnp_count; i++) {
        double dx = mnps[i].x - w->x;
        double dy = mnps[i].y - w->y;
        double dz = mnps[i].z - w->z;
        B += dipole_field(dx, dy, dz, mnps->M);
    }
    w->phase += B * 2 * M_PI * g * tau * 1e-3;
}

void setParameters(GPUData &d) {
    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);
    d.num_cells = num_cells;
    d.num_waters = num_water;
    d.timesteps = t;
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
    setParameters(d); // Copy over simulation constants for device

    // Seed the GPU random number generator
    d.seed = time(NULL) + rd();

    int totalThreads = num_blocks * threads_per_block;
    // Initialize a set of random states on the device
    HANDLE_ERROR(cudaMalloc((void**) &(d.states),
        totalThreads * sizeof(curandState_t)));

    // Allocate waters and copy over to the device
    HANDLE_ERROR(cudaMalloc((void **) &(d.waters),
        num_water * sizeof(water_info)));
    HANDLE_ERROR(cudaMemcpy((void*) waters, (void *) d.waters,
        sizeof(water_info) * num_water,
        cudaMemcpyHostToDevice));

    // Call the kernel and execute the simulation!
    simulateWaters<<<num_blocks, threads_per_block>>>(d);

    // Test copying the waters back to the host
    water_info* testWaters = new water_info[num_water];
    HANDLE_ERROR(cudaMemcpy((void*) testWaters, (void *) d.waters,
        sizeof(water_info) * num_water,
        cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_water; i++) {
        if((waters[i].x != testWaters[i].x)
            || (waters[i].y != testWaters[i].y)
            || (waters[i].z != testWaters[i].z))
            cout << "Memory Error at " << i << endl;
    }

    // Free used resources
    cudaFree(d.waters);
    cudaFree(d.states);

    delete[] linLattice;
    delete[] lookupTable;
    delete[] waters;
    delete mnps;
}
