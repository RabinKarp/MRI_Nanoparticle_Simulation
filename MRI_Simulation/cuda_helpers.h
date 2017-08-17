#ifndef CUDA_HELPERS
#define CUDA_HELPERS

#include <cstdio>
#include <vector>
#include "rand_walk.h"
#include "fcc_diffusion.h"
#include "octree.h"
#include "cuda.h"

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

//==============================================================================
// CUDA Utility functions
//==============================================================================

inline void* cudaAllocate(long long int size) {
    void *ptr;
    HANDLE_ERROR(cudaMalloc((void **) &ptr, size));
    return ptr;
}

inline void copyToDevice(void* dest, void* source, long long int size) {
    HANDLE_ERROR(cudaMemcpy(dest, source,
        size,
        cudaMemcpyHostToDevice));
}

inline void copyToHost(void* dest, void* source, long long int size) {
    HANDLE_ERROR(cudaMemcpy(dest, source,
        size,
        cudaMemcpyDeviceToHost));
}

#endif
