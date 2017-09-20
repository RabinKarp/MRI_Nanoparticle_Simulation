/**
 * @author  Vivek Bharadwaj 
 * @date    September 17, 2017
 * @file    gpu_random.cu 
 * @brief   Implementation details for the gpu_rng class. 
 */

#include "cuda.h"
#include "curand.h"
#include "gpu_random.h"

using namespace std;

/**
 * Initializes the GPU random number generator using the current time
 * as a seed.
 */
gpu_rng::gpu_rng() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    
    // Set the seed as the current time
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

/**
 * Destroys the GPU random number generator.
 */
gpu_rng::~gpu_rng() {
    curandDestroyGenerator(gen);
}

/**
 * Generates a specified number of random doubles in GPU memory
 * at the specified address sampled from a uniform distribution from
 * 0.0 to 1.0.
 *
 * Postcondition: The block of memory specified by the device pointer
 * is filled contiguously with the specified number of uniform random
 * doubles.
 *
 * @param n         An integer giving the number of doubles to generate
 * @param devData   A GPU device pointer to a block of memory to fill
 *                  with random doubles. The block of data specified
 *                  by the pointer must be at least
 *                  sizeof(double) * n bits in length.
 *
 */
void gpu_rng::getUniformDoubles(size_t n, double *devData)
{
    curandGenerateUniformDouble(gen, devData, n);
}

/**
 * Generates a specified number of random doubles in GPU memory
 * at the specified address sampled from a normal distribution with
 * standard deviation 1.0 
 *
 * Postcondition: The block of memory specified by the device pointer
 * is filled contiguously with the specified number of normal random
 * doubles.
 *
 * @param n         An integer giving the number of doubles to generate
 * @param devData   A GPU device pointer to a block of memory to fill
 *                  with random doubles. The block of data specified
 *                  by the pointer must be at least
 *                  sizeof(double) * n bits in length.
 *
 */
void gpu_rng::getNormalDoubles(size_t n, double *devData)
{
    curandGenerateNormalDouble(gen, devData, n, 0.0, 1.0);
}
