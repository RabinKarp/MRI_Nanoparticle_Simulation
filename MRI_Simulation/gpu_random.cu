
/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "curand.h"
#include "cuda_helpers.h"

using namespace std;

void getUniformDoubles(size_t n, double *devData)
{
    curandGenerator_t gen;;

    /* Create pseudo-random number generator */
    /*CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));*/
    curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_XORWOW);

    /* Set the seed as the current time*/
    curandSetPseudoRandomGeneratorSeed(gen,
                time(NULL));

    /* Generate n doubles on device */
    curandGenerateUniformDouble(gen, devData, n);

    /* Cleanup */
    curandDestroyGenerator(gen);
}


void getNormalDoubles(size_t n, double *devData)
{
    curandGenerator_t gen;;

    /* Create pseudo-random number generator */
    /*CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT));*/
    curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_XORWOW);

    /* Set the seed as the current time*/
    curandSetPseudoRandomGeneratorSeed(gen,
                time(NULL));

    /* Generate n doubles on device */
    curandGenerateNormalDouble(gen, devData, n, 0.0, 1.0);

    /* Cleanup */
    curandDestroyGenerator(gen);
}
