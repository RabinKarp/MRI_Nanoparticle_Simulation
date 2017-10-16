/*
 * @author  Aadyot Bhatnagar
 * @author  Vivek Bharadwaj
 * @date    August 15, 2017
 * @file    parameters.h
 * @brief   A file containing static const instantiations of all the paramaters
 *          that affect the way the simulation is conducted.
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <string>

#define STCONST static const

/* Switches for enabling or disabling debugging output files */
#undef DEBUG_LATTICE        // create output file w/ cell centers and neighbors?
#undef DEBUG_MNPS           // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?


struct ParameterStruct {
public:
    STCONST double g = 42.5781e6;             // gyromagnetic ratio in MHz/T
    
    #undef FULL_BOUNDARIES      // use full boundary conditions to calculate field?
    STCONST double scale = 7;     // calculate B explicitly within scale*R of clusterj 

    #ifndef FULL_BOUNDARIES     // otherwise, apply BC's at some cutoff distance
    STCONST double border = 6;    // boundary from box where we start applying BC's
    #endif

    STCONST int num_threads = 16; // number of CPU threads to run T2 simulation on

    /* Related to the CUDA kernel */
    #define threads_per_block 192 // Keep this as a multiple of 64

    // The variable below must be a multiple of the printing frequency
    STCONST int sprintSteps = 20000; // Each kernel execution handles AT MOST this many timesteps


    /* Molecule information; simulation performs at its peak when num_water is divisible by 64 */
    STCONST int num_water = 4032;             // number of waters in simulation

    /* Related to the cells in the simulation*/
    STCONST int num_cells = 270;               // Number of randomly thrown cells
    STCONST double cell_r = .55;                // cell radius in microns

    STCONST double mmoment = 3.5e-17;         // Magnetic moment for each cell

    // Exactly ONE of the two flags below must be set
    #undef CONSTANT_KICK
    #define RANDOM_KICK

    #ifdef CONSTANT_KICK 
    STCONST double phase_k = 2*3.14*42*12*5e-3;             // Intracellular ph. kick is k * dt at each tstep
    #elif defined RANDOM_KICK
    STCONST double phase_stdev = 2*3.14*1.5*1e-3*42*12*10e-3;         // St. dev. of intracellular phase accumulation
    STCONST double phase_k = 5;             //Chemical shift in ppm
    #endif

    /* Related to the simulation bounds */
    STCONST double bound = 30;                // full box is [0, bound]^3 (microns)

    /* All water molecules begin the simulation in a box with dimension
       water_start_bound^3 that is centered in the middle of the larger
       simulation box. Given in microns. */
    STCONST double water_start_bound = 10;

    /**
     * Define the flag below to force the simulation to avoid throwing water
     * molecules inside of cells initially. 
     */
    #define AVOID_INTRACELLULAR_THROW

    /* Parameters related to the optimized nearest cell finder */
    STCONST int hashDim = 20;
    STCONST int maxNeighbors = 13;

    /* Constants affecting diffusion */
    STCONST double D_cell = .5547;            // D in micron^2 per ms
    STCONST double D_extra = 1.6642;          // D in micron^2 per ms
    STCONST double P_expr = 0.2;             // permeability in micron per ms

    STCONST double tau = 1e-6; // Units of Microseconds
    
    /* Time scales and step sizes */        // tau defines time step in ms - currently must be power of 10
    STCONST int totaltime = 40;               // total time to run for in ms - because of GPU architecture, this
                                            // is constrained to be a discrete integer
    STCONST int t = (int)(totaltime/tau);     // Total time steps
    STCONST double taucp = 5.5;               // Carr-Purcell time in ms - up to 3 decimal places of precision 
    STCONST int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time

    
    // The following parameters are assigned values in the constructor below in this file 
    double reflectIO; // Assign values in constructor below, not here
    double reflectOI;

    // St. dev of displacements inside and outside cells 
    double in_stdev; // Assign values in constructor below, not here 
    double out_stdev;

    ParameterStruct(int dummy_flag) {
    /**
     * Each of the following doubles is in the range 0 to 1 and gives
     * the probability that a given water molecule will bounce off the cell boundary
     * attempting to diffuse from into the cell out of the cell (reflectIO) and the 
     * probability that a molecule bounces will diffusing out of the cell into the cell (reflectOI).
     * 
     * To make cells impermeable, set both of these numbers to 1. To make cell boundaries nonexistant,
     * set both numbers to 0.   
     */
        reflectIO = 0; // 1 - sqrt(tau / (6*D_cell)) * 4 * P_expr;
        reflectOI = 0; // 1 - ((1 - reflectIO) * sqrt(D_cell/D_extra));

        // St. dev of displacements inside and outside cells 
        in_stdev = sqrt(M_PI * D_cell * tau);
        out_stdev = sqrt(M_PI * D_extra * tau);
    }

    ParameterStruct() {
    }
}; 

extern ParameterStruct p;
const std::string delim = ",";          // Delimiter for output CSV file

#endif /* PARAMETERS_H */
