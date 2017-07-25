#ifndef GPU_RANDOM
#define GPU_RANDOM

#include "gpu_random.cu"
#include "cuda.h"
#include "curand.h"

/**
 * Returns a device pointer to an array of size n containing randomly generated
 * doubles.
 */

void getUniformDoubles(size_t n, double* devData);

void getNormalDoubles(size_t n, double* devData);

#endif
