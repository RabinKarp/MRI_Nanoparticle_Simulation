#ifndef GPU_RANDOM
#define GPU_RANDOM

#include "cuda.h"
#include "curand.h"

/**
 * Returns a device pointer to an array of size n containing randomly generated
 * doubles.
 */

class gpu_rng {
public:
    gpu_rng();
    ~gpu_rng();

    void getUniformDoubles(size_t n, double* devData);
    void getNormalDoubles(size_t n, double* devData);


private:
    curandGenerator_t gen;
};

#endif
