/*
 * @author  Aadyot Bhatnagar
 * @date    June 21, 2016
 * @file    rand_walk.cpp
 * @brief   This file includes functions to generate random displacement vectors
 *          and simulate the unbounded diffusion of water molecules.
 */

#include <random>
#include <cmath>
#include <iostream>
#include "rand_walk.h"
#include "fcc_diffusion.h"

/*
 * Returns an (x, y, z) displacement whose components all fall along the
 * normal distribution provided as an argument.
 */
water_info rand_displacement(normal_distribution<> normal, XORShift<> &gen)
{
    water_info w;
    w.x = normal(gen);
    w.y = normal(gen);
    w.z = normal(gen);
    return w;
}

/*
 * Given a set of Cartesian boundary conditions, initialize a given number of
 * water molecules distributed randomly over the cubical space [0, dim]^3.
 */
water_info *init_molecules(double dim, int num, vector<MNP_info> *mnps, XORShift<> &gen)
{
    water_info *molecules = new water_info[num];
    FCC lattice(cell_r, R_io, R_oi, 1000);
    water_info *temp = molecules;
    
    for (int i = 0; i < num; i++)
    {
        double x = gen.rand_pos_double() * dim;
        double y = gen.rand_pos_double() * dim;
        double z = gen.rand_pos_double() * dim;
        bool inside = false;

        // Make sure the molecule isn't inside a nanoparticle.
        vector<MNP_info>::iterator j;
        for (j = mnps->begin(); j < mnps->end() && !inside; j++)
        {
            double dx = x - j->x;
            double dy = y - j->y;
            double dz = z - j->z;
            if (NORMSQ(dx, dy, dz) < j->r * j->r)
                inside = true;
        }

        // If the molecule IS inside a nanoparticle, generate it again.
        if (inside)
        {
            i--;
            continue;
        }

        // Otherwise, store it in the array and move on.
        temp->x = x;
        temp->y = y;
        temp->z = z;
        lattice.update_nearest_cell_full(temp);
        temp++;
    }

    cout << "Molecules initialized!" << endl;
    return molecules;
}