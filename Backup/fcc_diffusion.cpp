/*
 * @author  Aadyot Bhatnagar
 * @date    June 22, 2016
 * @file    fcc_diffusion.cpp
 * @brief   Implementation details for a class that represents a 3x3x3 face-
 *          centered cubic lattice of cells with periodic boundary conditions.
 *          The only public method is one that simulates one step of the
 *          diffusion of a water molecule through the cell lattice.
 */

#include <cmath>
#include <iostream>
#include "fcc_diffusion.h"
#include "rand_walk.h"

#define CELL_TYPES 4

/*
 * Uses cell radius to scale coordinates of cell centers in FCC lattice and
 * initializes the scattering probabilities upon collision with a cell membrane,
 * both from the inside and from the outside. Also intializes the standard
 * deviation expected from the Gaussian distribution used to generate random
 * displacement vectors, as well as a random seed for the PRNG.
 */
FCC::FCC(double radius, double reflectIO, double reflectOI, int thread_id)
{
    for (int i = 0; i < 172; i++)
    {
        for (int j = 0; j < 3; j++)
            fcc[i][j] *= radius * sqrt(2);
    }

    this->reflectIO = reflectIO;
    this->reflectOI = reflectOI;
    this->in = sqrt(3 * pi * D_cell * tau);
    this->out = sqrt(3 * pi * D_extra * tau);
    this->radius = radius;
    this->bound = 3 * sqrt(2) * radius;

    for (int i = 0; i < thread_id; i++)
        gen.jump();
}

/*
 * Initializes the magnetic nanoparticle clusters in different cells of the
 * lattice. There is a 50% chance a cell has no MNPs, and an equal chance that
 * it is any of the other types. 
 */
vector<MNP_info> *FCC::init_mnps(XORShift<> &gen)
{
    /* Magnetic moments established by HD on NV adjusted using SQUID magneto-
     * metry for saturation magnetization. Each Cell is an array of the magnetic
     * moments of the enclosed nanoparticles.*/
    vector<double> cells[CELL_TYPES];
    cells[0] = {.0535e-12/2, .1688e-12/2};
    cells[1] = {0.0172e-11/2, 0.0850e-11/2, 0.1200e-11/2, 0.0026e-11/2,\
        0.0017e-11/2, 0.0017e-11/2, 0.0007e-11/2, 0.0019e-11/2};
    cells[2] = {0.0176e-12/2, 0.0023e-12/2, 0.0015e-12/2, 0.1531e-12/2};
    cells[3] = {0.3356e-11/2, 0.0617e-11/2, 0.1249e-11/2, 0.0197e-11/2};
    vector<MNP_info> *mnps = new vector<MNP_info>;

    for (int i = 0; i < NUM_CELLS; i++)
    {
        double coin = gen.rand_double();

        /* Give each different MNP distribution an equal chance of occupying
         * a given cell. There is a 50% chance coin < 0, so 50% chance cell j
         * remains unoccupied. */
        for (int j = CELL_TYPES - 1; j >= 0; j--)
        {
            if (coin > (double)j / (double)CELL_TYPES)
            {
                for (int k = 0; k < cells[j].size(); k++)
                {
                    double dist = gen.rand_pos_double() * radius;
                    double x = gen.rand_double();
                    double y = gen.rand_double();
                    double z = gen.rand_double();
                    double norm = NORMSQ(x, y, z);
                    x = (x / norm * dist) + fcc[i][0];
                    y = (y / norm * dist) + fcc[i][1];
                    z = (z / norm * dist) + fcc[i][2];

                    double M = cells[j][k];
                    double r = pow(M/(1e-15), 1.0/3.0) * mnp_radius;

                    if (x  < bound && x > -bound && y < bound && y > -bound &&
                        z < bound && z > -bound)
                        mnps->emplace_back(x, y, z, r, M);
                }
                break; // cell occupied -- don't try to fill it w/ more MNPs
            }
        }
    }

    return mnps;
}

/*
 * Determines the cell closest to the water molecule in question.
 */
void FCC::update_nearest_cell_full(water_info *w)
{
    /* Current position */
    double x = w->x;
    double y = w->y;
    double z = w->z;

    /* Determines distance from cell 0 to initialize data */
    int nearest = 0;
    double *center = fcc[0];
    double dx = x - center[0];
    double dy = y - center[1];
    double dz = z - center[2];
    double min_dist = NORMSQ(dx, dy, dz);

    /* Check distance to/from all neighboring cells */
    for (int i = 1; i < NUM_CELLS; i++)
    {
        center = fcc[i];
        dx = x - center[0];
        dy = y - center[1];
        dz = z - center[2]; 
        double curr_dist = NORMSQ(dx, dy, dz);
        if (curr_dist < min_dist)
        {
            min_dist = curr_dist;
            nearest = i;
        }
    }
    /* Update water molecule's record of nearest cell */
    w->nearest = nearest;
    if (in_cell(w))
        w->in_cell = true;
    else
        w->in_cell = false;
}

/*
 * For water molecules that have not crossed a boundary, this function updates 
 * the molecule's information about the cell it is closest to.
 */
void FCC::update_nearest_cell(water_info *w)
{
    /* Current position */
    double x = w->x;
    double y = w->y;
    double z = w->z;

    /* Information about nearest cell center on the last timestep */
    int nearest = w->nearest;
    int *check = neighbors[nearest];
    double *center = fcc[nearest];
    double dx = x - center[0];
    double dy = y - center[1];
    double dz = z - center[2];
    double min_dist = NORMSQ(dx, dy, dz);

    /* Check distance to/from all neighboring cells */
    for (int i = 0; i < NUM_NEIGHBORS; i++)
    {
        int curr_check = check[i];
        double *curr_center = fcc[curr_check];
        dx = x - curr_center[0];
        dy = y - curr_center[1];
        dz = z - curr_center[2];
        
        double curr_dist = NORMSQ(dx, dy, dz);
        if (curr_dist < min_dist)
        {
            min_dist = curr_dist;
            nearest = curr_check;
        }
    }

    /* Update water molecule's record of nearest cell */
    w->nearest = nearest;
}

/*
 * Determines if a molecule still resides in the cell it is associated with
 */
bool FCC::in_cell(water_info *w)
{
    double *center = fcc[w->nearest];
    double x = w->x - center[0];
    double y = w->y - center[1];
    double z = w->z - center[2];
    return radius * radius > NORMSQ(x, y, z);
}

/*
 * Determines whether or not the water molecule has crossed a boundary. The
 * appropriate additions are applied if it has.
 */
bool FCC::boundary_conditions(water_info *w)
{
    bool cross = false;

    if (w->x > bound)
    {
        w->x -= 2 * bound;
        w->cross_x++;
        cross = true;
    }
    else if (w->x < -bound)
    {
        w->x += 2 * bound;
        w->cross_x--;
        cross = true;
    }

    if (w->y > bound)
    {
        w->y -= 2 * bound;
        cross = true;
    }
    else if (w->y < -bound)
    {
        w->y += 2 * bound;
        cross = true;
    }

    if (w->z > bound)
    {
        w->z -= 2 * bound;
        cross = true;
    }
    
    else if (w->z < -bound)
    {
        w->z += 2 * bound;
        cross = true;
    }

    return cross;
}

/*
 * Simulates one step of diffusion for a single water molecule, w. Also
 * accounts for reflections off of magnetic nanoparticles. Returns the node of
 * the octree the water is resident in at the end of the step.
 */
oct_node *FCC::diffusion_step(water_info *w, Octree *tree)
{
    water_info disp;
    int old_nearest = w->nearest;
    oct_node *voxel = tree->get_voxel(w);

    // check for reflection off a cell membrane if w starts inside a cell
    if (w->in_cell)
    {
        disp = rand_displacement(norm_in, gen);
        *w += disp;
        update_nearest_cell(w);

        if (!in_cell(w))
        {
            if (gen.rand_pos_double() < reflectIO)
            {
                *w -= disp;
                w->nearest = old_nearest;
                return voxel;
            }
            else
                w->in_cell = false;
        }
    }

    // check for reflection off a cell membrane if w starts outside a cell
    else
    {
        disp = rand_displacement(norm_out, gen);
        *w += disp;
        update_nearest_cell(w);

        if (in_cell(w))
        {
            if (gen.rand_pos_double() < reflectOI)
            {
                *w -= disp;
                w->nearest = old_nearest;
                return voxel;
            }
            else
                w->in_cell = true;
        }
    }

    // check for reflection off a magnetic nanoparticle
    vector<MNP_info> *mnps = tree->mnps;
    for (vector<MNP_info>::iterator i = mnps->begin(); i < mnps->end(); i++)
    {
        double dx = w->x - i->x;
        double dy = w->y - i->y;
        double dz = w->z - i->z;
        double r = i->r;
        if (NORMSQ(dx, dy, dz) < r * r)
        {
            *w -= disp;
            w->nearest = old_nearest;
            if (in_cell(w))
                w->in_cell = true;
            else
                w->in_cell = false;
            return voxel;
        }
    }

    // Account for periodic boundary conditions
    if (boundary_conditions(w))
        update_nearest_cell_full(w);

    return tree->get_voxel(w);
}
