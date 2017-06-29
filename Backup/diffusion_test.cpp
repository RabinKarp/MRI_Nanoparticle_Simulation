/*
 * @author  Aadyot Bhatnagar
 * @date    June 29, 2016
 * @file    diffusion_test.cpp
 * @brief   Outputs a .csv-formatted output of the square displacements (x
 *          direction) of all the molecules being simulated at every time step.
 *          Use the terminal to generate a .csv.
 */

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "fcc_diffusion.h"

#define NUM_WATER 10000

extern const double tau, cell_r, R_io, R_oi;
const double bound = 3 * sqrt(2) * cell_r;
extern const int t;
using namespace std;

/*
 * Simulates unbounded diffusion over a specified number of time steps and
 * prints out the square displacement of each water molecule over this time
 * period.
 */
int main(void)
{
    XORShift<> gen(time(NULL));
    vector<MNP_info> empty;
    water_info *molecules = init_molecules(bound, NUM_WATER, &empty, gen);
    FCC *lattice = new FCC(cell_r, R_io, R_oi, 1);
    double max_product = 1e-7, max_g = 1, min_g = .5;
    Octree *tree = new Octree(max_product, max_g, min_g, 2, 1);
    water_info *temp = molecules;

    ofstream out_file;
    string filename("diffusion_test.csv");
    out_file.open(filename);
    
    for (int i = 0; i < t; i++)
    {
        temp = molecules;
        for (int j = 0; j < NUM_WATER; j++)
        {
            oct_node *voxel = tree->get_voxel(temp);
            if (i % 10000 == 0)
                out_file << temp->x + temp->cross_x * bound * 2 << ",";
            double dx = temp->x - voxel->x;
            double dy = temp->y - voxel->y;
            double dz = temp->z - voxel->z;
            double g = voxel->g;
            assert(dx*dx < g*g && dy*dy < g*g && dz*dz < g*g);
            lattice->diffusion_step(temp, tree);
            temp++;
        }
        out_file << endl;
    }

    out_file.close();
    delete lattice;
    delete tree;
    delete[] molecules;
    return 0;
}
