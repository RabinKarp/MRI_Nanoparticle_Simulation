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

const extern double cell_r;

int main(void)
{
    int n = 2;
    double tree_size = 0, max_size = 0, power = 12;
    double bound = 3 * sqrt(2) * cell_r;
    cout << bound << endl;
    double max_product = 1e-4, max_g = 1, min_g = bound / (pow(2, power) - 1);

    Octree *tree = new Octree(max_product, max_g, min_g, 0, n);
    cout << max_product << " used for max_product." << endl;

    for (int i = 0; i < (int)pow(8, n); i++)
        tree_size += (double)tree->space[i]->size();
    for (int i = n; i <= (int)power; i++)
        max_size += pow(8, i);
    double pct = tree_size / max_size * 100;
    cout << "The tree has " << tree_size << " nodes." << endl;
    cout << "Max possible " << max_size << " nodes." << endl;
    cout << "Used " << pct << "% of max possible space." << endl;

    delete tree;
    return 0;
}
