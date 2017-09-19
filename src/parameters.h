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

#define HIGH        5000000
#define MAX_MNPS    1000    // Upper bound to number of cells in the simulation (modify as needed)

/* Parameters affecting nanoparticle residency in nodes */
#undef FULL_BOUNDARIES      // use full boundary conditions to calculate field?
const double raw_scale = 2; // calculate B explicitly within scale*R of cluster
const double scale = raw_scale;

#ifndef FULL_BOUNDARIES     // otherwise, apply BC's at some cutoff distance
const double border = 6;    // boundary from box where we start applying BC's
#endif

/* Parameters affecting the T2 simulation */
const int num_threads = 16; // number of CPU threads to run T2 simulation on

/* Switches for enabling or disabling debugging output files */
#undef DEBUG_LATTICE        // create output file w/ cell centers and neighbors?
#undef DEBUG_MNPS           // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?

/* Related to the CUDA kernel */
#define threads_per_block 192 // Keep this as a multiple of 64

// The variable below must be a multiple of the printing frequency
const int sprintSteps = 20000; // Each kernel execution handles AT MOST this many timesteps


/* Molecule information; simulation performs at its peak when num_water is divisible by 64 */
const int num_water = 4032;             // number of waters in simulation

/* Related to the cells in the simulation*/
const int num_cells = 257;               // Number of randomly thrown cells
const double cell_r = .55;                // cell radius in microns

const double mmoment = 8.1957e-18;         // Magnetic moment for each cell

// Exactly ONE of the two flags below must be set
#define CONSTANT_KICK
#undef RANDOM_KICK

#ifdef CONSTANT_KICK 
const double phase_k = 2*3.14*42*7*2.1e-3;             // Intracellular ph. kick is k * dt at each tstep
#elif defined RANDOM_KICK
const double phase_stdev = 1.0;         // St. dev. of intracellular phase accumulation
const double phase_k = 2*3.14*42*7*2.1e-3;
#endif

/* Related to the simulation bounds */
const double bound = 20;                // full box is [0, bound]^3 (microns)

/* All water molecules begin the simulation in a box with dimension
   water_start_bound^3 that is centered in the middle of the larger
   simulation box. Given in microns. */
const double water_start_bound = 10;

/**
 * Define the flag below to force the simulation to avoid throwing water
 * molecules inside of cells initially. 
 */
#undef AVOID_INTRACELLULAR_THROW

/* Parameters related to the optimized nearest cell finder */
const int hashDim = 20;
const int maxNeighbors = 13;

/* Constants affecting diffusion */
const double D_cell = .5547;            // D in micron^2 per ms
const double D_extra = 1.6642;          // D in micron^2 per ms
const double P_expr = 0.2;             // permeability in micron per ms

/**
 * Each of the following doubles is in the range 0 to 1 and gives
 * the probability that a given water molecule will bounce off the cell boundary
 * attempting to diffuse from into the cell out of the cell (reflectIO) and the 
 * probability that a molecule bounces will diffusing out of the cell into the cell (reflectOI).
 * 
 * To make cells impermeable, set both of these numbers to 1. To make cell boundaries nonexistant,
 * set both numbers to 0.   
 */

const double tau = 1e-6;
const double reflectIO = 1 - sqrt(tau / (6*D_cell)) * 4 * P_expr;
const double reflectOI = 1 - ((1 - reflectIO) * sqrt(D_cell/D_extra));

/* Time scales and step sizes */        // tau defines time step in ms - currently must be power of 10
const int totaltime = 40;               // total time to run for in ms - because of GPU architecture, this
                                        // is constrained to be a discrete integer
const int t = (int)(totaltime/tau);     // Total time steps
const double taucp = 5.5;               // Carr-Purcell time in ms - up to 3 decimal places of precision 
const int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time

const std::string delim = ",";          // Delimiter for output CSV file

#endif /* PARAMETERS_H */
