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
 * Returns an (x, y, z) displacement whose norm is equal to a specified d.
 */
water_info rand_displacement(double d, XORShift<> &gen)
{
    double dx = gen.rand_double();
    double dy = gen.rand_double();
    double dz = gen.rand_double();
    double norm = sqrt(NORMSQ(dx, dy, dz));
    
    water_info w;
    w.x = dx / norm * d;
    w.y = dy / norm * d;
    w.z = dz / norm * d;
    return w;
}

/*
 * Simulates unbounded diffusion, given a diffusion length L and a number of
 * time steps.
 */
void unbounded_diffusion(water_info *molec, int steps, int num, double L,\
        XORShift<> &gen)
{
    std::normal_distribution<> norm(0, L);
    for (water_info *temp = molec; temp < molec + num; temp++)
    {
        for (int j = 0; j < steps; j++)
        {
            water_info disp = rand_displacement(norm(gen), gen);
            *temp += disp;
        }
    }
}