/**
 * @author  Vivek Bharadwaj 
 * @date    September 17, 2017
 * @file    FCCBox.cpp 
 * @brief   Implementation details for the FCCBox class. 
 */

#include "FCCBox.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>

#include "parameters.h"
#include "rand_walk.h"

using namespace std;

/**
 * Initializes the BacteriaBox by calling the superclass SimulationBox
 * constructor. populateSimulation() must be called after construction
 * to populate the box with its components.
 * 
 * @param gen           A pointer to the XORShift random number generator
 *                      used to initialize the MNPs and waters
 */
FCCBox::FCCBox(XORShift<> *gen)
    :
    SimulationBox(gen)
{
    // Empty
}

/**
 * Destroys the BacteriaBox entirely through the superclass destructor.
 */
FCCBox::~FCCBox() {
    // Derived class has no special resources that we need to clean up
}

/**
 * Initializes cells within the simulation bound by throwing them in the
 * configuration of a face-centered cubic lattice.
 */
void FCCBox::init_cells() {    
    for (int i = 0; i < 172; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fcc[i][j] *= p.cell_r * sqrt(2)*p.fcc_pack; 
            fcc[i][j] += p.bound/2;
        }
        cells.emplace_back(fcc[i][0], fcc[i][1], fcc[i][2]); 
    }             
}

/**
 * Initalizes the waters in the simulation by initializing them in a sub-cube
 * of dimension WATER_START_BOUND centered within the larger simulation cube.
 * When the flag AVOID_INTRACELLULAR_THROW is set, the simulation avoids
 * throwing waters inside cels initially.
 */
void FCCBox::init_waters() {
    water_info current;

    double offset = (p.bound - p.water_start_bound) / 2.0;
    for (int i = 0; i < p.num_water; i++) { 
        bool invalid = true;

        // Re-throw the water molecule repeatedly until we get a valid one
        while(invalid) {
            invalid = false; 
            current.x = offset + gen->rand_pos_double() * p.water_start_bound;
            current.y = offset + gen->rand_pos_double() * p.water_start_bound;
            current.z = offset + gen->rand_pos_double() * p.water_start_bound;

#ifdef AVOID_INTRACELLULAR_THROW
            // When the appropriate flag is defined, re-throw if inside a cell
            invalid = 
                (checkLatticeContainment(current.x, current.y, current.z) >= 0);
#endif

            // Here, check for containment within MNPs
			// If a water molecule is contained by an MNP, re-throw it 
			for(int i = 0; i < mnps.size(); i++) {
				double dx = mnps[i].x - current.x;
				double dy = mnps[i].y - current.y;
				double dz = mnps[i].z - current.z;

				if(NORMSQ(dx, dy, dz) < mnps[i].r) {
					invalid = true;
				}	
			}
        } 
        current.phase = 0; 
        waters.push_back(current); 
    }

    std::cout << "Molecules initialized!" << std::endl;
}

water_info FCCBox::rand_displacement(double d, XORShift<> *gen) {
	double dx = gen->rand_double();
    double dy = gen->rand_double();
    double dz = gen->rand_double();
    double norm = sqrt(NORMSQ(dx, dy, dz));

    water_info w;
    w.x = dx / norm * d;
    w.y = dy / norm * d;
    w.z = dz / norm * d;
    return w;
}

#ifdef UNCLUSTERED
/*
* Initializes the specified number of individaul, unclustered magnetic
* nanoparticles all in extracellular space.
*/
void FCCBox::init_mnps() {
{	
	for (int i = 0; i < p.num_mnps; i++)
	{
		double x, y, z;
		bool invalid = true;

		// keep generating (x,y,z) coordinates until we get an extracellular one
		// that doesn't overlap with other nanoparticles
		while (invalid)
		{
			x = gen->rand_pos_double() * p.bound;
			y = gen->rand_pos_double() * p.bound;
			z = gen->rand_pos_double() * p.bound;
			invalid = false;

			// re-throw if the nanoparticle is inside/outside a cell (depends on flag)
			for (int j = 0; j < p.num_cells && !invalid; j++)
			{
				double dx = x - fcc[j][0];
				double dy = y - fcc[j][1];
				double dz = z - fcc[j][2];

				// Check for invalid MNP's (overlapping with cell boundaries,
				// not in desired intracellular or extracellular locations, and
				// set a flag to rethrow if needed)
#if defined EXTRACELLULAR
				if (NORMSQ(dx, dy, dz) < pow(p.cell_r + p.mnp_radius, 2)) {
					invalid = true;
				}
#elif defined INTRACELLULAR
				if ((checkLatticeContainment(x, y, z) == -1) ||
					checkLatticeOverlap(x, y, z, p.mnp_radius)) {
					invalid = true;
				}
				// Just check for cell boundary overlap in this case
#elif defined INTRA_EXTRA
				if ((NORMSQ(dx, dy, dz) > pow(p.cell_r - p.mnp_radius, 2))
					&& (NORMSQ(dx, dy, dz) < pow(p.cell_r + p.mnp_radius, 2)))
					invalid = true;
#endif
			}

			// re-throw if the nanoparticle overlaps with another nanoparticle
			std::vector<MNP_info>::iterator curr;
			for (curr = mnps.begin(); curr != mnps.end() && !invalid; curr++)
			{
				double dx = x - curr->x;
				double dy = y - curr->y;
				double dz = z - curr->z;
				if (NORMSQ(dx, dy, dz) < pow(2 * p.mnp_radius, 2))
					invalid = true;
			}
		}

		// If the flag is set, factor in the new lipid envelope
		double radius = p.mnp_radius;
#ifdef LIPID_ENVELOPE
		if (checkLatticeContainment(x, y, z) != -1)
			radius += lipid_width;
#endif
		double mmoment = p.mmoment;
		mnps.emplace_back(x, y, z, radius, mmoment);
	}

}
/* endif UNCLUSTERED */
#elif defined CLUSTERED
/**
 * Initializes an MNP (really a dipole for the cell) at the center of each cell
 * with the specified magnetic moment and 0 radius. 0 radius implies that
 * water molecules are free to diffuse as close as they like to the dipole,
 * so it is necessary to subtract off field contributions when waters are
 * inside cells.
 */
void FCCBox::init_mnps() {
	/* Magnetic moments established by HD on NV adjusted using SQUID magneto-
     * metry for saturation magnetization. Each Cell is an array of the magnetic
     * moments of the enclosed nanoparticles.*/
	
	int CELL_TYPES = 6;
    std::vector<double> cells[CELL_TYPES];
    cells[0] = {5.35e-14/3.33, 1.688e-13/3.33};
    cells[1] = {1.72e-13/3.33, 8.50e-13/3.33, 1.200e-12/3.33, 2.6e-14/3.33,\
        1.7e-14/3.33, 1.7e-14/3.33, 7e-15/3.33, 1.9e-14/3.33};
    cells[2] = {1.76e-14/3.33, 2.3e-15/3.33, 1.5e-15/3.33, 1.531e-13/3.33};
    cells[3] = {1.126e-15*3.33, 4.68e-16*3.33};
    cells[4] = {2.129e-15*3.33, 3.653e-15*3.33};
    cells[5] = {2.096e-15*3.33,1.68e-15*3.33,2.3e-16*3.33};
    //cells[6] = {0.3356e-11/3.33, 0.0617e-11/3.33, 0.1249e-11/3.33, 0.0197e-11/3.33};
   

    std::uniform_real_distribution<> dist(1 - (1/p.prob_labeled), 1);
    for (int i = 0; i < p.num_cells; i++)
    {
        double coin = dist(*gen);

        /* Give each different MNP distribution an equal chance of occupying
         * a given cell. There is a preset chance that the cell is labeled at
         * all. */
        for (int j = CELL_TYPES - 1; j >= 0; j--)
        {
            if (coin > (double) j / (double) CELL_TYPES)
            {
                for (unsigned k = 0; k < cells[j].size(); k++)
                {
                    double M = cells[j][k];
                    double r = pow(M/(1.6e-15), 1.0/3.0) * p.mnp_radius;
                    double x, y, z;

                    /* Keep re-generating the center for the MNP in question
                     * until the MNP does not overlap with any other MNPs that
                     * have already been initialized. */
                    bool invalid = true;
                    while (invalid)
                    {
                        invalid = false;
#ifdef INTRACELLULAR
                        double norm = gen->rand_pos_double() * (p.cell_r);
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];
#elif defined EXTRACELLULAR
ifdef INTRACELLULAR
  double norm = gen->rand_pos_double() * (p.cell_r);
 water_info loc = rand_displacement(norm, gen);
 x = loc.x + fcc[i][0];
 y = loc.y + fcc[i][1];
 z = loc.z + fcc[i][2];
#ifdef THROW_FREE // Throw cluster anywhere in extracellular space - the check
                  // against cell containment occurs after the branch
                        x = gen->rand_pos_double() * p.bound;
                        y = gen->rand_pos_double() * p.bound;
                        z = gen->rand_pos_double() * p.bound;

#else // Throw the particle within a certain vicinity of the labeled cell
                        double norm = p.cell_r * (1 + gen->rand_pos_double()
                            * (u_throw_coeff - 1));
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];
#endif
                        // Re-throw in case of cell containment, for both cases
                        // THROW_FREE set and unset
                        if(checkLatticeContainment(x, y, z) != -1)
                            invalid = true;
#elif defined INTRA_EXTRA
                        double norm = gen->rand_pos_double() * p.cell_r * u_throw_coeff;
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];

                        // If the MNP center occurs inside ANOTHER cell,
                        // check and re-throw
                        if(checkLatticeContainment(x, y, z) != -1 &&
                            checkLatticeContainment(x, y, z) != i)
                            invalid = true;
#endif
						// Check against overlap with other MNPs
                        std::vector<MNP_info>::iterator m, start = mnps.begin();
                        for (m = start; m != mnps.end() && !invalid; m++)
                        {
                            double dx = x - m->x;
                            double dy = y - m->y;
                            double dz = z - m->z;
                            if (NORMSQ(dx, dy, dz) < pow(r + m->r, 2))
                                invalid = true;
                        }
                    }

                    /* Only actually place MNP in lattice if its center is
                     * inside the defined space (so we don't artificially
                     * increase MNP density via periodic boundary conditions) */
                    if (x  < p.bound && x > 0 && y < p.bound && y > 0 &&
                        z < p.bound && z > 0) {
#ifdef LIPID_ENVELOPE
                        if(checkLatticeContainment(x, y, z) != -1)
                            r += lipid_width;
#endif
                        mnps.emplace_back(x, y, z, r, M);
                    }
                }
                break; // cell occupied -- don't try to fill it w/ more MNPs
            }
        }
	} 

#endif
	}
