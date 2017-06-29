/*
 * @author  Aadyot Bhatnagar
 * @date    June 30, 2016
 * @file    octree_test.cpp
 * @brief   Builds an octree and deletes it. Reports information about how
 *          space is being used (i.e. what portion of the max theoretical
 *          size the data structure occupies). Test can be timed to subtract
 *          off as an offset to analyze the efficiency of the T2 simulation.
 */

#include <iostream>
#include "octree.h"
#include "fcc_diffusion.h"
#include "parameters.h"

int main(void)
{
    uint64_t p = 8;
    double max_product = -1, max_g = bound/.9, min_g = bound / (pow(2, p) - .1);

    XORShift<> gen(time(NULL));
    std::cerr << "Using " << max_product << " for max_product and " << scale;
    std::cerr << " for scale." << std::endl;
    FCC lattice(D_cell, D_extra, P_expr);
    std::vector<MNP_info> *mnps = lattice.init_mnps(gen);
    Octree *tree = new Octree(max_product, max_g, min_g, gen, mnps);
    
    delete tree;
    return 0;
}