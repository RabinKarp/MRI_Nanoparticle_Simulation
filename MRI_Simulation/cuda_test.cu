#define HIGH           5000000
#define MAX_MNPS       1000
#define M_PI           3.14159265358979323846

#include "cuda_helpers.h"

#include <stdio.h>
#include <time.h>
#include <dos.h>
#include <windows.h>
#include <stdlib.h>

#include <vector>
#include <fstream>
#include <thread>
#include <iostream>
#include "math.h"
#include "parameters.h"
#include "fcc_diffusion.h"
#include "gpu_random.h"
#include "octree.h"

using namespace std;
/**
 *  nvcc cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -arch=sm_61 -lcurand -ccbin "
 * C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64_x86"
 */

#define threads_per_block 256
const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T

// Each kernel execution handles AT MOST this many timesteps
const int sprintSteps = 10000;
const int num_uniform_doubles = 4; // # of uniform doubles per water per tstep
const int num_normal_doubles = 1;  // # of normal  doubles per water per tstep

__constant__ Triple dev_lattice[num_cells];

//==============================================================================

/**
 * Initialize a GPU verison of the octree from the CPU version.
 */
void initOctree(Octree *oct, GPUData &d) {
    // Initialize octree parameters

    vector<oct_node> *vec_nodes = oct->space;

    gpu_node *localTree = new gpu_node[vec_nodes->size()];
    MNP_info *localMNPs = new MNP_info[oct->mnps->size()];

    int mnp_idx = 0;
    for(int i = 0; i < vec_nodes->size(); i++) {
        oct_node current = (*vec_nodes)[i];
        localTree[i].mc = current.mc;

        for(int j = 0; j < 8; j++) {
            localTree[i].child[j] = current.child[j];
        }

        localTree[i].numResidents = current.resident->size();
        localTree[i].resIdx = mnp_idx;

        for(int j = 0; j < current.resident->size(); j++) {
            localMNPs[mnp_idx] = (*(current.resident))[j];
            mnp_idx++;
        }
    }

    HANDLE_ERROR(cudaMalloc((void **) &(d.tree),
        vec_nodes->size() * sizeof(gpu_node)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.mnps),
        oct->mnps->size() * sizeof(MNP_info)));

    // Allocate arrays for morton codes
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_x),
        256 * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_y),
        256 * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_z),
        256 * sizeof(uint32_t)));

    HANDLE_ERROR(cudaMemcpy(d.tree, localTree,
        vec_nodes->size() * sizeof(gpu_node),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.mnps, localMNPs,
        oct->mnps->size() * sizeof(MNP_info),
        cudaMemcpyHostToDevice));

    // Copy over the Morton code arrays
    HANDLE_ERROR(cudaMemcpy(d.morton_x, morton_x,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.morton_y, morton_y,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.morton_z, morton_z,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    delete[] localTree;
    delete[] localMNPs;
}

__device__ double dipole_field(double dx, double dy, double dz, double M)
{
    double d2x = dx*dx;
    double d2y = dy*dy;
    double d2z = dz*dz;
    double sum = d2x + d2y + d2z;
    double divisor = sum * sum * sqrt(sum);
    return M * 1e11 * (2*d2z - d2x - d2y) / divisor;
}

__device__ int morton_code(int depth, double x, double y, double z, GPUData &d) {

    double size = pow(2.0, depth);
    uint32_t idx_x = floor(x / d.bound * size);
    uint32_t idx_y = floor(y / d.bound * size);
    uint32_t idx_z = floor(z / d.bound * size);
    uint64_t answer = 0;
    // start by shifting the third byte, since we only look @ the first 21 bits
    if (depth > 16)
    {
        answer |=   d.morton_z[(idx_z >> 16) & 0xFF ] |
                    d.morton_y[(idx_y >> 16) & 0xFF ] |
                    d.morton_x[(idx_x >> 16) & 0xFF ];
        answer <<= 24;
    }

    // shift second byte
    if (depth > 8)
    {
        answer |=   d.morton_z[(idx_z >> 8) & 0xFF ] |
                    d.morton_y[(idx_y >> 8) & 0xFF ] |
                    d.morton_x[(idx_x >> 8) & 0xFF ];
        answer <<= 24;
    }

    // shift first byte
    answer |=   d.morton_z[(idx_z) & 0xFF ] |
                d.morton_y[(idx_y) & 0xFF ] |
                d.morton_x[(idx_x) & 0xFF ];
    return answer;
}

__device__ gpu_node* find_tree(double wx, double wy, double wz, GPUData &d)
{
    return d.tree + morton_code(d.min_depth, wx, wy, wz, d);
}

/*
 * Helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 */
__device__ unsigned find_child(double wx, double wy, double wz, int d, GPUData &data)
{
    return morton_code(d + 1, wx, wy, wz, data) & 7;
}

// TODO: Check this function!
__device__ gpu_node* find_node(gpu_node *n, double wx, double wy, double wz, int d, GPUData &data) {
    while( ! (n->mc >> 63)) {
        // otherwise, navigate to the appropriate child and recurse
        unsigned child_no = find_child(wx, wy, wz, d, data);
        d += 1;
        n += n->child[child_no].idx;
    }
    return n;
}

__device__ gpu_node* get_voxel(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    return find_node(find_tree(wx, wy, wz, d), wx, wy, wz, d.min_depth, d);
}

/**
 * Returns the B field at the location of a particular water molecule
 */
__device__ double get_field(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    gpu_node *leaf = get_voxel(w, d);

    uint64_t depth = 0, mc = (leaf->mc << 1) >> 1;
    while (mc >>= 3) depth++;

    // use Morton code's depth to find child index to find value of B to return
    unsigned child_no = find_child(wx, wy, wz, depth, d);
    double B = (double)leaf->child[child_no].B;

    // add in contributions from resident MNPs zeroed out during construction
    for(int i = leaf->resIdx; i < leaf->resIdx + leaf->numResidents; i++) {
        MNP_info *np = d.mnps + i;
        B += dipole_field(wx - np->x, wy - np->y, wz - np->z, np->M);
    }

    return B;
}

//==============================================================================

void finalizeGPU(GPUData &d) {
    cudaFree(d.waters);
    cudaFree(d.flags);
    cudaFree(d.uniform_doubles);
    cudaFree(d.normal_doubles);
    cudaFree(d.magnetizations);
    cudaFree(d.mnps);
    cudaFree(d.tree);

    cudaFree(d.morton_x);
    cudaFree(d.morton_y);
    cudaFree(d.morton_z);
}

void setParameters(GPUData &d) {
    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);

    d.reflectIO = 1 - sqrt(tau / (6*D_cell)) * 4 * P_expr;
    d.reflectOI = 1 - ((1 - d.reflectIO) * sqrt(D_cell/D_extra));

    d.num_cells = num_cells;
    d.num_waters = num_water;
    d.timesteps = sprintSteps;
    d.cell_r = cell_r;
    d.bound = bound;
    d.nBlocks = num_blocks;
    d.g = g;
    d.tau = tau;
    d.bound = bound;
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

__device__ bool cell_reflect(water_info *i, water_info *f, int tStep, GPUData &d) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double coin = d.uniform_doubles[tStep * d.num_waters * 4 + tid * 4 + 3];
    bool flip = (i->in_cell && (! f->in_cell) && coin < d.reflectIO)
                    || ((! i->in_cell) && f->in_cell && coin < d.reflectOI);
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

    if(w->in_cell) {
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
    w->x = fmod(w->x + d.bound, d.bound);
    w->y = fmod(w->y + d.bound, d.bound);
    w->z = fmod(w->z + d.bound, d.bound);
}

__device__ void accumulatePhase(water_info *w, GPUData &d) {
    double B = get_field(w, d);
    w->phase += B * 2 * M_PI * d.g * d.tau * 1e-3;
}
// END PHASE ACCUMULATION FUNCTIONS

__device__ void carrPurcellFlip(water_info *w, int tStep) {

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
        w = d.waters[tid];
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
            //updateNearest(&w, d);

            // Check cell boundary / MNP reflection
            if(cell_reflect(&init, &w, i, d))
                w = init;

            accumulatePhase(&w, d);
            carrPurcellFlip(&w, i);

            /*
            // Copy the magnetization to shared memory
            mags[i] = cos(w.phase);

            // Perform a memory reduction
            sumMagnetizations(mags, i, d.magnetizations);
            */
        }
    }

    if(tid < d.num_waters) {
        d.flags[tid] = x;
        d.waters[tid] = w;
    }
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

    // Initialize the octree
    double max_product = 2e-6, max_g = 5, min_g = .002;
    uint64_t sTime = time(NULL);
    Octree tree(max_product, max_g, min_g, gen, mnps);
    uint64_t eTime = time(NULL) - sTime;
    std::cout << "Octree took " << eTime / 60 << ":";
    if (eTime % 60 < 10) std::cout << "0";
    std::cout << eTime % 60 << " to build." << std::endl << std::endl;

    GPUData d;
    setParameters(d);
    d.num_mnps = mnps->size();
    //initOctree(&tree, d);

    cout << "Allocated GPU Octree!" << endl;
    int totalUniform =  num_uniform_doubles * num_water * sprintSteps;
    int totalNormal = num_normal_doubles * num_water * sprintSteps;
    // Allocations: Perform all allocations here
    HANDLE_ERROR(cudaMalloc((void **) &(d.waters),
        num_water * sizeof(water_info)));
    HANDLE_ERROR(cudaMalloc((void **) &(dev_lattice),
        num_cells * sizeof(Triple)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.flags),
        num_water * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &d.uniform_doubles,
        totalUniform*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &d.normal_doubles,
        totalNormal*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.magnetizations),
        num_blocks * t * sizeof(double)));

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
    for(int i = 0; i < 10; i++) {
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
