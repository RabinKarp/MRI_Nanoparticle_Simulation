/**
 * @author  Vivek Bharadwaj
 * @date    18 September 2017.
 * @file    cuda_sim.h 
 * @brief   Contains a single public function definition to simulate diffusion
 *          and phase kicks experienced by molecules within a simulation box
 *          with waters, semi-permeable cell boundaries, and magnetic dipoles.
 */

#ifndef SIM_HEADER
#define SIM_HEADER

#include <cstring>
#include <vector>
#include "parameters.h"
#include "rand_walk.h"
#include "fcc_diffusion.h"
#include "octree.h"
#include "cuda.h"

using namespace std;

/**
 * The gpu_node struct is the GPU analogue of the CPU struct oct_node.
 * It contains a 64 bit morton code, a set of child nodes, a specified
 * number of resident MNPs, and a pointer to an array of MNP residents.
 */
struct gpu_node {
    uint64_t mc;        // Morton code of node; leaf if leftmost bit is set
    B_idx child[8];     // child offsets (internal) or child B fields (leaves)
    int numResidents;
    MNP_info* resident;
} gpu_node;

/**
 * The GPUData struct contains pointers to the arrays required by the
 * GPU kernels to perform the diffusion simulations, as well as
 * all relevant simulation parameters.
 */
typedef struct GPUData {
    // Related to the GPU
    int nBlocks;

    // Related to the octree
    int num_mnps;
    int min_depth;
    int max_depth;
    gpu_node** __restrict__ tree;
    gpu_node** localPointers;
    int *sizes;
    int arr_size;

    std::vector<void*> *addresses;

    // Morton code arrays
    uint32_t* __restrict__ morton_x;
    uint32_t* __restrict__ morton_y;
    uint32_t* __restrict__ morton_z;

    // Related to the lattice
    int num_cells;
    double cell_r;
    double bound;
    Triple* lattice;
    int hashDim;
    int** __restrict__ lookupTable;
    int** __restrict__ localLookup;

    // Related to diffusion
    double reflectIO;
    double reflectOI;
    double in_stdev;
    double out_stdev;

    // Related to simulation time
    int tcp;
    int pfreq;
    int* time;

    // Related to the waters
    int num_waters;
    water_info* waters;

    // Physical constants
    double g;
    double tau;

#ifdef RANDOM_KICK
    double phase_stdev;
#elif defined CONSTANT_KICK
    double phase_k;
#endif

    // Related to the GPU's random number resources
    double *x, *y, *z; 
    double *coins;
    double *normal_doubles;

    bool* in_cell;

    long long int timesteps;

     // Memory for debugging purposes only
     int *flags;
} GPUData;


void simulateWaters(string fName);

#endif
