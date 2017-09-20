/**
 * @author  Vivek Bharadwaj 
 * @author  Aadyot Bhatnagar
 * @date    September 17, 2017
 * @file    SimulationBox.h 
 * @brief   Header file for the simulation box class, which initializes
 *          water molecules, semi-permeable cells, MNPs, and the octree. 
 */


#ifndef SIM_BOX_HEADER
#define SIM_BOX_HEADER

#include "rand_walk.h"
#include "octree.h"

using namespace std;

/**
 * A Triple is a transparent data structure containing x, y, and z coordinates
 * and a constructor to initialize the structure with 3 supplied values.
 */
struct Triple {
    double x;
    double y;
    double z;

public:
    Triple(double xVal, double yVal, double zVal):
        x(xVal), 
        y(yVal), 
        z(zVal)
        {}
};

/**
 * Keeps track of all relevant objects in the simulation box, including the
 * semi-permeable cells, water molecule starting positions, MNPs, and the octree.
 * This class is abstract so that subclasses can implement their own unique
 * mechanisms for cell initialization (random throw vs. lattice), MNP
 * initialization, and water initialization.
 *
 * After calling the constructor for any subclass, clients must call the
 * method populateSimulation() to generate the waters, MNPs, and cells
 * and to initialize the octree. Public functions are defined to retrieve
 * pointers or references to each of the simulation components. Errors
 * result by attempting to retrieve simulation components without first
 * calling populateSimulation().
 */
class SimulationBox {
public:
    SimulationBox(int num_cells, int num_waters, XORShift<> *gen);
    virtual ~SimulationBox();

    void populateSimulation();

    // Public access functions
    Triple*     getCells();
    water_info* getWaters(); 
    MNP_info*   getMNPs();
    int**       getLookupTable(); 
    Octree*     getOctree();

    int         getMNPCount();

    // Debugging
    void print_simulation_stats();

protected:
    XORShift<> *gen;

    bool populated; 
    int num_cells;
    int num_waters;
    int num_intra_mnps;

    Octree*             tree;
    vector<MNP_info>    mnps;
    vector<water_info>  waters;
    vector<Triple>      cells;
    int**               lookupTable;

    bool checkLatticeOverlap(double x, double y, double z, double r);
    int checkLatticeContainment(double x, double y, double z);

private:
    virtual void init_cells()       = 0;
    virtual void init_mnps()        = 0;
    virtual void init_waters()      = 0;
    
    void init_lookuptable(); 
    void apply_bcs_on_mnps();

    void sortWaters();
};

#endif
