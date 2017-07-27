#ifndef CUDA_HELPERS
#define CUDA_HELPERS

#include <cstdio>
#include "rand_walk.h"
#include "octree.h"

// Macro for GPU-callable member functions
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef struct gpu_node {
    uint64_t mc;        // Mocton code of node; leaf if leftmost bit is set
    B_idx child[8];     // child offsets (internal) or child B fields (leaves)
    int numResidents;
    int resIdx;
} gpu_node;

typedef struct GPUData {
    // Related to the GPU
    int nBlocks;

    // Related to the octree
    int num_mnps;
    int min_depth;
    MNP_info *mnps;
    gpu_node *tree;

    // Morton code arrays
    int* morton_x;
    int* morton_y;
    int* morton_z;

    // Related to the lattice
    int num_cells;
    double cell_r;
    double bound;
    int* sphereLookup;

    // Related to diffusion
    double reflectIO;
    double reflectOI;
    double in_stdev;
    double out_stdev;

    // Related to the waters
    int num_waters;
    water_info* waters;

    // Physical constants
    double g;
    double tau;

    // Related to the GPU's random number resources
    double* uniform_doubles;
    double* normal_doubles;

    /**
     * The array of magnetizations is a double array of dimension
     * (t * num_blocks). Each block writes to a unique portion of the shared
     * global memory.
     */
     double* magnetizations;
     long long int timesteps;

     // Memory for debugging purposes only
     int *flags;
} GPUData;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );

        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#endif
