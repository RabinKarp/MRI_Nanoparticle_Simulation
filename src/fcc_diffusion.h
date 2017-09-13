/*
 * @author  Aadyot Bhatnagar
 * @date    August 10, 2016
 * @file    fcc_diffusion.h
 * @brief   Header file outlining a class that represents a 3x3x3 face-centered
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
#include "rand_walk.h"
#include "octree.h"

struct Triple {
    double x;
    double y;
    double z;
};

/*
 * This class encodes a 3x3x3 face-centered cubic lattice with periodic
 * boundary conditions. These cell boundaries are used to initialize magnetic
 * nanoparticles and water molecules throughout the lattice.
 */
class FCC
{
    public:
    FCC(double D_in, double D_out, double P, XORShift<> &gen);
    ~FCC();

    double fcc[num_cells][3];

    void initializeCells(XORShift<> &gen);
    void initializeLookupTable();

    void sortWaters(water_info* waters, int num_water, Octree &tree);

    double diffusion_step(water_info *w, Octree *tree, XORShift<> &gen);
    std::vector<MNP_info> *init_mnps();
    water_info *init_molecules(int n, XORShift<> &gen);
    std::vector<MNP_info> *init_cluster(MNP_info &init, double r_pack,\
        int num_mnp, XORShift<> &gen);
    
    void update_nearest_cell_full(water_info *w);
    Triple* linearLattice();
    void update_nearest_cell(water_info *w);

    int** lookupTable;

    private:
    double reflectIO, reflectOI;
    std::normal_distribution<> norm_in, norm_out;

    bool checkLatticeOverlap(double x, double y, double z, double r);
    int checkLatticeContainment(double x, double y, double z);

    bool in_cell(water_info *w);
    bool boundary_conditions(water_info *w);
    void print_mnp_stats(std::vector<MNP_info> *mnps);
    void apply_bcs_on_mnps(std::vector<MNP_info> *mnps);
};

#endif /* FCC_DIFFUSION_H */
