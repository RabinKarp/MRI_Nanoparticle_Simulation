
/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */

#include "cuda.h"
#include "curand.h"
#include "gpu_random.h"

using namespace std;

gpu_rng::gpu_rng() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    
    /* Set the seed as the current time*/
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

}

gpu_rng::~gpu_rng() {
    curandDestroyGenerator(gen);
}

void gpu_rng::getUniformDoubles(size_t n, double *devData)
{
    curandGenerateUniformDouble(gen, devData, n);
}


void gpu_rng::getNormalDoubles(size_t n, double *devData)
{
    curandGenerateNormalDouble(gen, devData, n, 0.0, 1.0);
}
