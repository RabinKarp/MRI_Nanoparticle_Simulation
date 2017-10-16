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
#include "BacteriaBox.h"
#include "octree.h"
#include "cuda.h"

using namespace std;

/**
 * The gpu_node struct is the GPU analogue of the CPU struct oct_node.
 * It contains a 64 bit morton code, a set of child nodes, a specified
 * number of resident MNPs, and a pointer to an array of MNP residents.
 */
typedef struct gpu_node {
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
    // Related to the octree
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
    Triple* lattice; 
    int** __restrict__ lookupTable;
    int** __restrict__ localLookup;

	int** __restrict__ mnpLookupTable;
	int** __restrict__ mnpLocalLookup;

	int* time;

    // Related to the waters
    water_info* waters;

    // Related to the GPU's random number resources
    double *x, *y, *z; 
    double *coins;
    double *normal_doubles;

    bool* in_cell;
} GPUData;


void simulateWaters(string fName);

#endif
