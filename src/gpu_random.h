/**
 * @author  Vivek Bharadwaj 
 * @date    September 17, 2017
 * @file    gpu_random.h 
 * @brief   Header file for the gpu_rng class, a host class that generates
 *          random numbers on the GPU and keeps track of a curandGenerator_t
 *          object.
 */

#ifndef GPU_RANDOM
#define GPU_RANDOM

#include "cuda.h"
#include "curand.h"

/**
 * Encapsulates methods for generating random numbers on the GPU. The
 * class keeps track of a curandGenerator_t object that is destroyed
 * when the object goes out of scope. The class can generate random doubles
 * from both a uniform distribution and a normal distribution with standard
 * deviation 1 (simply multiply by the outputs by any other standard
 * deviation to simulate sampling from any other distribution.
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
