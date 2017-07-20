/*
 * @author  Aadyot Bhatnagar
 * @date    August 10, 2016
 * @file    fcc_diffusion.h
 * @brief   Header file outlining a class that represents a n x n x n face-cent.
 *          cubic lattice of cells. This file contains most of the parameters
 *          that affect the diffusion behavior and specific calculations used
 *          in the T2 simulation.
 */

#ifndef FCC_DIFFUSION_H
#define FCC_DIFFUSION_H

#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include "parameters.h"
#include "rand_walk.h"
#include "octree.h"

/* Intrinsic properties of a face-centered cubic lattice of spheres */
const int num_cells =               // number of cells in the FCC lattice
    1 + (3 * n) + (6 * n * n) + (4 * n * n * n);
const int num_neighbors = 12;       // number of neighbors each FCC cell has

/**
 * x, y, and z offsets for neighbors of a given cell in FCC lattice
 */
const int nOffsets[12][3] = {
  {1, 0, 1}, {0, 1, 1}, {-1, 0, 1}, {0, -1, 1},
  {1, 1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, -1, 0},
  {1, 0, -1}, {0, 1, -1}, {-1, 0, -1}, {0, -1, -1}
};

/*
 * This class encodes a n x n x n face-centered cubic lattice with periodic
 * boundary conditions. These cell boundaries are used to initialize magnetic
 * nanoparticles and water molecules throughout the lattice.
 */
class FCC
{
    public:
    FCC(double D_in, double D_out, double P);
    double diffusion_step(water_info *w, Octree *tree, XORShift<> &gen);
    std::vector<MNP_info> *init_mnps(XORShift<> &gen);
    water_info *init_molecules(double L, int n, std::vector<MNP_info> *mnps,\
        XORShift<> &gen);
    std::vector<MNP_info> *init_cluster(MNP_info &init, double r_pack,\
        int num_mnp, XORShift<> &gen);
    void update_nearest_cell_full(water_info *w);

    private:
    double reflectIO, reflectOI;
    std::normal_distribution<> norm_in, norm_out;

    void initLattice(int dim);
    bool in_cell(water_info *w);
    bool boundary_conditions(water_info *w);
    void update_nearest_cell(water_info *w);
    void print_mnp_stats(std::vector<MNP_info> *mnps);
    void apply_bcs_on_mnps(std::vector<MNP_info> *mnps);

    /*
     * Instance variable representing the centers of all the cells in an FCC
     * lattice (unscaled).
     */
    double fcc[num_cells][3];

    /*
     * Instance variable where the array stored at the ith index corresponds to
     * the list of all the neighbors of the ith cell in the fcc array above.
     */
    int neighbors[num_cells][num_neighbors];
};

#endif /* FCC_DIFFUSION_H */
