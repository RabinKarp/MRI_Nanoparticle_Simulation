/**
 * @author  Vivek Bharadwaj 
 * @author  Aadyot Bhatnagar
 * @date    September 17, 2017
 * @file    SimulationBox.cpp 
 * @brief   Implementation details for the Simulation Box class keeping
 *          track of all of the simulation components.
 */

#include "SimulationBox.h"
#include "octree.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Initializes a SimulationBox with the specified number of cells and
 * water molecules, using the specified random number generator. Calling the
 * constructor allocates memory for the simulation objects, but does not
 * actually initialize them - call populateSimulation() to actually populate
 * the simulation.
 *
 * @param rng           A pointer to the XORShift random number generator
 *                      to generate the waters, cells, and MNPs
 */
SimulationBox::SimulationBox(XORShift<> *rng) {
    gen = rng; 

    // Allocate memory for the lookup table
    lookupTable = new int*[p.hashDim * p.hashDim * p.hashDim];
    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        lookupTable[i] = new int[p.maxNeighbors]; 
    }

    mnpLookupTable = new int*[p.hashDim * p.hashDim * p.hashDim];
    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        mnpLookupTable[i] = new int[p.maxNeighbors]; 
    }
    
}

/**
 * Destroys all components of the simulation - the cell lookup table
 * is manually cleaned up, and the octree is deleted.
 */
SimulationBox::~SimulationBox() {
    if(populated)
        delete tree;  

    // Delete the lookup table
    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        delete[] (lookupTable[i]);
    }
    delete[] lookupTable;

    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        delete[] (mnpLookupTable[i]);
    }
    delete[] mnpLookupTable;

}

/**
 * Populates the components of the simulation, allowing the client
 * to access cells, waters, MNPs, the cell lookup table, and the octree.
 *
 * The simulation is constructed in the following manner:
 *
 * 1. Cells are constructed with a call to init_cells().
 * 2. MNPs are constructed with a call to init_mnps().
 * 3. Waters are constructed with a call to init_waters().
 * 4. A fast cell lookup table (spatial map) is constructed with a call to
 *    init_lookuptable(). 
 * 5. Periodic boundary conditions are applied to the MNPs in the simulation
 *    box.
 * 6. The octree is initialized from the MNP configuration.
 * 7. The water molecules are sorted by the morton codes of their starting
 *    positions - this produces a serious performance improvement without
 *    affecting any other aspect of the simulation.
 *
 * Postcondition: The simulation is now populated and the client can access
 *                the simulation components via the accessors.
 */
void SimulationBox::populateSimulation() {
    init_cells();    
    init_mnps();
    num_intra_mnps = mnps.size();

    init_waters();
    
    init_lookuptable();
    init_MNPlookuptable();
    apply_bcs_on_mnps(); 

    // Initialize the octree for fast field computation
    double max_product = 2e-6, max_g = 5, min_g = .002;
    uint64_t sTime = time(NULL);
    tree = new Octree(max_product, max_g, min_g, *gen, &mnps);
    uint64_t eTime = time(NULL) - sTime;
    std::cout << "Octree took " << eTime / 60 << ":";
    if (eTime % 60 < 10) std::cout << "0";
    std::cout << eTime % 60 << " to build." << std::endl << std::endl;
    // End octree initialization
    
    sortWaters();
    populated = true;
}

/**
 * Structure used to sort water molecules by the morton codes of their
 * initial positions.
 */
struct SortStruct {
    water_info w;
    uint64_t mc;
};

/**
 * Helper comparison function used to sort water molecules on the morton
 * codes of their initial positions.
 *
 * @param a Structure containing first water molecule and MC to compare
 * @param b Structure containing second water molecule and MC to compare
 *
 * @return true If the first water molecule has a morton code less than
 *              the second water molecule.
 */
bool compare(SortStruct &a, SortStruct &b) {
    return a.mc < b.mc;
}

/**
 * Sorts the water molecules in the simulation on the basis of the
 * morton codes of their initial positions. The algorithm creates
 * an array of SortStruct objects and loads a water molecule into each one,
 * then uses the octree to compute the morton code of the octree voxel
 * containing each molecule. The data structures are then sorted based
 * on morton code, and the water molecules are copied back from the
 * SortStruct array to the simulation box's array of water molecules.
 */
void SimulationBox::sortWaters() {
    SortStruct* objs = new SortStruct[p.num_water];

    for(int i = 0; i < p.num_water; i++) {
        objs[i].w = waters[i];
        oct_node* voxel = tree->get_voxel(&(objs[i].w));
        objs[i].mc = voxel->mc; 
    }

    sort(objs, objs + p.num_water, compare);

    for(int i = 0; i < p.num_water; i++) {
        waters[i] = objs[i].w;
    }

    delete[] objs; 
}

/**
* Helper function that checks if the given (x, y, z) coordinate is
* contained in any cell in the simulation box. Returns the index of the lattice
* cell that contains the given point, or -1 if no cell contains that point.
*
* @param x X coordinate to check
* @param y Y coordinate to check
* @param z Z coordinate to check
*
* @return An integer index (>= 0) of the lattice cell containing the given
*         point, or -1 if no cell contains that point.
*/
int SimulationBox::checkLatticeContainment(double x, double y, double z) {
    int containCell = -1;
    for(int i = 0; i < p.num_cells; i++) {
        double dx = x - cells[i].x;
        double dy = y - cells[i].y;
        double dz = z - cells[i].z;

        if(NORMSQ(dx, dy, dz) < pow(p.cell_r, 2)) {
            containCell = i;
        }
    }
    return containCell;
}

/**
 * Helper function that checks if the sphere with the given (x, y, z)
 * coordinates and radius OVERLAPS with the boundary of any sphere in
 * the cellList lattice.
 *
 * @param x X coordinate of sphere center to check
 * @param y Y coordinate of sphere center to check
 * @param z Z coordinate of sphere center to check
 * @param r The radius of the input sphere
 *
 * @return true If the given sphere overlaps with any other sphere in the
 *              simulation, false otherwise.
 */
bool SimulationBox::checkLatticeOverlap(double x, double y, double z, double r) {
    bool overlaps = false;
    
    for(int i = 0; i < p.num_cells; i++) {
        double dx = x - cells[i].x;
        double dy = y - cells[i].y;
        double dz = z - cells[i].z;

        double sqDist = NORMSQ(dx, dy, dz);

        if (sqDist < pow(p.cell_r + r, 2)
            && sqDist > pow((p.cell_r - r), 2)) {
            overlaps = true;
        }
    }
    return overlaps;
}

/**
 * Prints the state of the simulation box out to a file. The components printed
 * are:
 *
 * 1. A list of intra-box MNPs, their respective positions and radii
 * 2. A list of water molecules and their respective positions
 * 3. A list of cells, their respective positions and radii.  
 */
void SimulationBox::print_simulation_stats() {
    assert(populated);

    cout << "Printing simulation statistics..." << endl;

    ofstream fout("simulation_stats.csv");
    fout << "Number of MNPs, " << num_intra_mnps << endl;
    for(auto it = mnps.begin(); it < mnps.begin() + num_intra_mnps; it++) {
        fout << (*it).x << "," << (*it).y << "," << (*it).z << "," << (*it).r 
            << endl;
    }

    fout << "Number of Waters, " << p.num_water << endl;
    for(auto it = waters.begin(); it < waters.end(); it++) {
        fout << (*it).x << "," << (*it).y << "," << (*it).z << endl;
    }

    fout << "Number of Cells, " << p.num_cells << endl;
    for(auto it = cells.begin(); it < cells.end(); it++) {
        fout << (*it).x << "," << (*it).y << "," << (*it).z << "," << p.cell_r 
            << endl;
    }
    fout.close();
}

/**
 * Initializes a fast cell lookup table for use in the simulation. The cell
 * lookup table considers a lattice of dimension hashDim * hashDim * hashDim
 * of equally spaced points within the simulation box. At each of these
 * points, the method loops through all of the cells in the simulation
 * and finds all those within a distance cell_r + the diagonal of the unit
 * cube defining the lattice. The entry for that particular lattice point
 * is updated with a list of (at most) MAX_NEIGHBORS candidate cells whose 
 * center is a distance cell_r + diagonal away from the lattice point.
 *
 * When a water molecule wishes to determine whether it is inside a cell
 * or not, it can simply look up which lower left-corner lattice point
 * it is closest to. The list of closest neighbor cells for this lattice
 * point provides a small (~10-13) list of candidates for cells the
 * water could be in, speeding up the computation.
 *
 * The lookup tables are  2D integer array of dimension hashDim * hashDim
 * * hashDim * hashDim by MAX_NEIGHBORS. In other words, each lattice
 * point keeps track of MAX_NEIGHBORS indices of possible nearest neighbors.
 * There will be fewer than MAX_NEIGHBORS cell indices within the
 * given distance of the lattice point, so any remaining unfilled entires
 * for lattice pt. are filled with the sentinel value -1.
 */
void SimulationBox::init_lookuptable() {
    double cubeLength = p.bound / p.hashDim;
    double diagonal = sqrt(3) * cubeLength;

    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        vector<int> ncells;
        double x = (i % p.hashDim) * cubeLength;
        double y = ((i / p.hashDim) % (p.hashDim)) * cubeLength;
        double z = i / (p.hashDim * p.hashDim) * cubeLength;

        for(int j = 0; j < p.num_cells; j++) {
            double dx = x - cells[j].x;
            double dy = y - cells[j].y;
            double dz = z - cells[j].z;
            if(sqrt(NORMSQ(dx, dy, dz)) < p.cell_r + diagonal) {
                ncells.push_back(j);
            }
        }
     
        assert(ncells.size() < p.maxNeighbors);

        for(int j = 0; j < p.maxNeighbors; j++) {
            if(j < ncells.size())
                lookupTable[i][j] = ncells[j];
            else
                lookupTable[i][j] = -1;
        }
    }
}

void SimulationBox::init_MNPlookuptable() {
    double cubeLength = p.bound / p.hashDim;
    double diagonal = sqrt(3) * cubeLength;

    for(int i = 0; i < p.hashDim * p.hashDim * p.hashDim; i++) {
        vector<int> ncells;
        double x = (i % p.hashDim) * cubeLength;
        double y = ((i / p.hashDim) % (p.hashDim)) * cubeLength;
        double z = i / (p.hashDim * p.hashDim) * cubeLength;

        for(int j = 0; j < num_intra_mnps; j++) {
            double dx = x - mnps[j].x;
            double dy = y - mnps[j].y;
            double dz = z - mnps[j].z;
            if(sqrt(NORMSQ(dx, dy, dz)) < mnps[j].r + diagonal) {
                ncells.push_back(j);
            }
        }

        assert(ncells.size() < p.maxNeighbors);

        for(int j = 0; j < p.maxNeighbors; j++) {
            if(j < ncells.size())
                mnpLookupTable[i][j] = ncells[j];
            else
                mnpLookupTable[i][j] = -1;
        }
    }
}

int** SimulationBox::getMNPLookupTable() {
    return mnpLookupTable;
}


#ifdef FULL_BOUNDARIES
/**
 * Duplicates existing nanoparticles initialized in the lattice across all
 * periodic boundaries.
 */
void SimulationBox::apply_bcs_on_mnps()
{
    unsigned init_size = mnps->size();
    for (unsigned i = 0; i < init_size; i++)
    {
        MNP_info *np = mnps.data() + i;
        double x = np->x, y = np->y, z = np->z, r = np->r, M = np->M;

        // boundaries touching faces
        mnps.emplace_back(x - p.bound, y, z, r, M);
        mnps.emplace_back(x - p.bound, y, z, r, M);
        mnps.emplace_back(x, y - p.bound, z, r, M);
        mnps.emplace_back(x, y + p.bound, z, r, M);
        mnps.emplace_back(x, y, z - p.bound, r, M);
        mnps.emplace_back(x, y, z + p.bound, r, M);

        // boundaries touching edges
        mnps.emplace_back(x, y - p.bound, z - p.bound, r, M);
        mnps.emplace_back(x, y - p.bound, z + p.bound, r, M);
        mnps.emplace_back(x, y + p.bound, z - p.bound, r, M);
        mnps.emplace_back(x, y + p.bound, z + p.bound, r, M);

        mnps.emplace_back(x - p.bound, y, z - p.bound, r, M);
        mnps.emplace_back(x - p.bound, y, z + p.bound, r, M);
        mnps.emplace_back(x + p.bound, y, z - p.bound, r, M);
        mnps.emplace_back(x + p.bound, y, z + p.bound, r, M);

        mnps.emplace_back(x - p.bound, y - p.bound, z, r, M);
        mnps.emplace_back(x - p.bound, y + p.bound, z, r, M);
        mnps.emplace_back(x + p.bound, y - p.bound, z, r, M);
        mnps.emplace_back(x + p.bound, y + p.bound, z, r, M);

        // boundaries touching corners
        mnps.emplace_back(x - p.bound, y - p.bound, z - p.bound, r, M);
        mnps.emplace_back(x - p.bound, y - p.bound, z + p.bound, r, M);
        mnps.emplace_back(x - p.bound, y + p.bound, z - p.bound, r, M);
        mnps.emplace_back(x - p.bound, y + p.bound, z + p.bound, r, M);
        mnps.emplace_back(x + p.bound, y - p.bound, z - p.bound, r, M);
        mnps.emplace_back(x + p.bound, y - p.bound, z + p.bound, r, M);
        mnps.emplace_back(x + p.bound, y + p.bound, z - p.bound, r, M);
        mnps.emplace_back(x + p.bound, y + p.bound, z + p.bound, r, M);
    }
}

#else
/*
 * Of the nanoparticles initially initialized, if some are within a pre-set
 * range of the p.border of the space being simulated, those nanoparticles will
 * be duplicated across the appropriate periodic p.boundaries.
 */
void SimulationBox::apply_bcs_on_mnps()
{
    unsigned init_size = mnps.size();
    for (unsigned i = 0; i < init_size; i++)
    {
        MNP_info *np = mnps.data() + i;
        double x = np->x, y = np->y, z = np->z, r = np->r, M = np->M;

        if (x + p.border > p.bound) // near front side
        {
            mnps.emplace_back(x - p.bound, y, z, r, M);

            if (y + p.border > p.bound) // near front and right sides
            {
                mnps.emplace_back(x - p.bound, y - p.bound, z, r, M);
                mnps.emplace_back(x, y - p.bound, z, r, M);

                if (z + p.border > p.bound) // near front, right, and top sides
                {
                    mnps.emplace_back(x - p.bound, y - p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x - p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y - p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near front, right, and bottom sides
                {
                    mnps.emplace_back(x - p.bound, y - p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x - p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y - p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else if (y - p.border < 0) // near front and left sides
            {
                mnps.emplace_back(x - p.bound, y + p.bound, z, r, M);
                mnps.emplace_back(x, y + p.bound, z, r, M);

                if (z + p.border > p.bound) // near front, left, and top sides
                {
                    mnps.emplace_back(x - p.bound, y + p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x - p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y + p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near front, left, and bottom sides
                {
                    mnps.emplace_back(x - p.bound, y + p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x - p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y + p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else // not near left or right sides, but near front side
            {
                if (z + p.border > p.bound) // near front and top sides
                {
                    mnps.emplace_back(x - p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }
                else if (z - p.border < 0) // near front and bottom sides
                {
                    mnps.emplace_back(x - p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }
        }

        else if (x - p.border < 0) // near back side
        {
            mnps.emplace_back(x + p.bound, y, z, r, M);

            if (y + p.border > p.bound) // near back and right sides
            {
                mnps.emplace_back(x + p.bound, y - p.bound, z, r, M);
                mnps.emplace_back(x, y - p.bound, z, r, M);

                if (z + p.border > p.bound) // near back, right, and top sides
                {
                    mnps.emplace_back(x + p.bound, y - p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x + p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y - p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near back, right, and bottom sides
                {
                    mnps.emplace_back(x + p.bound, y - p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x + p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y - p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else if (y - p.border < 0) // near back and left sides
            {
                mnps.emplace_back(x + p.bound, y + p.bound, z, r, M);
                mnps.emplace_back(x, y + p.bound, z, r, M);

                if (z + p.border > p.bound) // near back, left, and top sides
                {
                    mnps.emplace_back(x + p.bound, y + p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x + p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y + p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near back, left, and bottom sides
                {
                    mnps.emplace_back(x + p.bound, y + p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x + p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y + p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else // not near left or right sides, but near back side
            {
                if (z + p.border > p.bound) // near back and top sides
                {
                    mnps.emplace_back(x + p.bound, y, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }
                else if (z - p.border < 0) // near back and bottom sides
                {
                    mnps.emplace_back(x + p.bound, y, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }
        }

        else // not near back or front sides
        {
            if (y + p.border > p.bound) // near right side
            {
                mnps.emplace_back(x, y - p.bound, z, r, M);

                if (z + p.border > p.bound) // near top and right sides
                {
                    mnps.emplace_back(x, y - p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near bottom and right sides
                {
                    mnps.emplace_back(x, y - p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else if (y - p.border < 0) // near left side
            {
                mnps.emplace_back(x, y + p.bound, z, r, M);

                if (z + p.border > p.bound) // near top and left sides
                {
                    mnps.emplace_back(x, y + p.bound, z - p.bound, r, M);
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                }

                else if (z - p.border < 0) // near bottom and left sides
                {
                    mnps.emplace_back(x, y + p.bound, z + p.bound, r, M);
                    mnps.emplace_back(x, y, z + p.bound, r, M);
                }
            }

            else // not near left, right, front, or back sides
            {
                if (z + p.border > p.bound) // near top side
                    mnps.emplace_back(x, y, z - p.bound, r, M);
                else if (z - p.border < 0) // near bottom side
                    mnps.emplace_back(x, y, z + p.bound, r, M);
            }
        }
    }
}
#endif

/**
 * Returns a pointer to an array containing the water molecules
 * initialized by the simulation.
 *
 * @return A pointer to an array of water molecules initialized by the box
 */
water_info* SimulationBox::getWaters() {
    assert(populated);
    return waters.data();
}

/**
 * Returns a pointer to an array containing the MNPs (including those
 * duplicated across periodic boundary conditions) initialized by the
 * simulation.
 *
 * @return A pointer to an array of MNPs
 */
MNP_info* SimulationBox::getMNPs() {
    assert(populated);
    return mnps.data();
}

/**
 * Returns a pointer to a pointer to an integer representing the nearest cell 
 * lookup table initialized by the simulation box.
 *
 * @return A pointer to the nearest cell lookup table
 */
int** SimulationBox::getLookupTable() {
    assert(populated);
    return lookupTable;
}

/**
 * Returns a pointer to the octree generated from the MNP configuration within
 * this simulation box.
 *
 * @return A pointer to the octree generated by the simulation box.
 */
Octree* SimulationBox::getOctree() {
    assert(populated);
    return tree;
}

/**
 * Returns a pointer to an array of cells maintained by the simulation box.
 *
 * @return A pointer to an array of cells maintained by the simulation box.
 */
Triple* SimulationBox::getCells() {
    assert(populated);
    return cells.data();
}

/**
 * Returns the number of MNPs tracked by the simulation box. This is NOT the 
 * same as the number of intra-box MNPs, as the simulation also tracks
 * nanoparticles that have been duplicated across periodic boundary conditions.
 *
 * @return The number of MNPs tracked by the simulation box
 */
int SimulationBox::getMNPCount() {
    return mnps.size(); 
}
