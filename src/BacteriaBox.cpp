/**
 * @author  Vivek Bharadwaj 
 * @date    September 17, 2017
 * @file    BacteriaBox.cpp 
 * @brief   Implementation details for the BacteriaBox class. 
 */

#include "BacteriaBox.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>

#include "parameters.h"
#include "rand_walk.h"

using namespace std;

/**
 * Initializes the BacteriaBox by calling the superclass SimulationBox
 * constructor. populateSimulation() must be called after construction
 * to populate the box with its components.
 * 
 * @param num_cells     The number of cells in the simulation box
 * @param num_waters    The number of waters in the simulation box
 * @param gen           A pointer to the XORShift random number generator
 *                      used to initialize the MNPs and waters
 */
BacteriaBox::BacteriaBox(int num_cells, int num_waters, XORShift<> *gen)
    :
    SimulationBox(num_cells, num_waters, gen)
{
    // Only action: the superclass constructor is called with an single
    // magnetic dipole for each cell
}

/**
 * Destroys the BacteriaBox entirely through the superclass destructor.
 */
BacteriaBox::~BacteriaBox() {
    // Derived class has no special resources that we need to clean up
}

/**
 * Initializes cells within the simulation bound by randomly throwing them
 * and checking that they don't overlap with each other within the simulation
 * bound. If cells do overlap, they are simply re-thrown.
 */
void BacteriaBox::init_cells() {
    for(int i = 0; i < num_cells; i++) {
        bool invalid = true;
        double x, y, z;
        while(invalid) {
            invalid = false;
            x = cell_r + gen->rand_pos_double() * (bound - 2 * cell_r);
            y = cell_r + gen->rand_pos_double() * (bound - 2 * cell_r);
            z = cell_r + gen->rand_pos_double() * (bound - 2 * cell_r);

            // Check against overlap with other cells
            for(int j = 0; j < i; j++) {
                double dx = cells[j].x - x;
                double dy = cells[j].y - y;
                double dz = cells[j].z - z;
                
                // Use 4 * cell_r^2 because separation must be >= 2* cell_r
                if(NORMSQ(dx, dy, dz) < 4 * cell_r * cell_r)
                    invalid = true;
            }
        }
        cells.emplace_back(x, y, z);
    }
}

/**
 * Initalizes the waters in the simulation by initializing them in a sub-cube
 * of dimension WATER_START_BOUND centered within the larger simulation cube.
 * When the flag AVOID_INTRACELLULAR_THROW is set, the simulation avoids
 * throwing waters inside cels initially.
 */
void BacteriaBox::init_waters() {
    water_info current;

    double offset = (bound - water_start_bound) / 2.0;
    for (int i = 0; i < num_waters; i++) { 
        bool invalid = true;

        // Re-throw the water molecule repeatedly until we get a valid one
        while(invalid) {
            invalid = false; 
            current.x = offset + gen->rand_pos_double() * water_start_bound;
            current.y = offset + gen->rand_pos_double() * water_start_bound;
            current.z = offset + gen->rand_pos_double() * water_start_bound;

#ifdef AVOID_INTRACELLULAR_THROW
            // When the appropriate flag is defined, re-throw if inside a cell
            invalid = 
                (checkLatticeContainment(current.x, current.y, current.z) >= 0);
#endif
        } 
        current.phase = 0; 
        waters.push_back(current); 
    }

    std::cout << "Molecules initialized!" << std::endl;
}

/**
 * Initializes an MNP (really a dipole for the cell) at the center of each cell
 * with the specified magnetic moment and 0 radius. 0 radius implies that
 * water molecules are free to diffuse as close as they like to the dipole,
 * so it is necessary to subtract off field contributions when waters are
 * inside cells.
 */
void BacteriaBox::init_mnps() {
    num_intra_mnps = num_cells;
    for(int i = 0; i < num_cells; i++) {
        mnps.emplace_back(cells[i].x, cells[i].y, cells[i].z, 0,
            mmoment);
    } 
}
