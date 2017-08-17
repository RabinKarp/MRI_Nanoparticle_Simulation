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

/* Parameters affecting nanoparticle residency in nodes */
#undef FULL_BOUNDARIES      // use full boundary conditions to calculate field?
const double raw_scale = 7; // calculate B explicitly within scale*R of cluster
const double scale = raw_scale;

#ifndef FULL_BOUNDARIES     // otherwise, apply BC's at some cutoff distance
const double border = 6;    // boundary from box where we start applying BC's
#endif

/* Parameters affecting the T2 simulation */
#undef EXPLICIT             // calculate B field explicitly? *DEPRECATED
const int num_threads = 16; // number of threads to run T2 simulation on
const int num_runs = 1;     // number of times to run T2 simulation

/* Switches for enabling or disabling debugging output files */
#undef DEBUG_LATTICE        // create output file w/ cell centers and neighbors?
#undef DEBUG_DIFF          // create output file w/ RMS displacements?
#undef DEBUG_MNPS           // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?
#undef TIMED_OUTPUT         // print out a progress report every 1ms?

/* Related to the CUDA kernel */
#define threads_per_block 128
const int sprintSteps = 25000; // Each kernel execution handles AT MOST this many timesteps


/* Molecule and nanoparticle info */
const int num_water = 1;             // number of waters in simulation

/* Related to the cells in the simulation*/
const int num_cells = 1;               // Number of randomly thrown cells
const double cell_r = 2;                // cell radius in microns

const double mmoment = 1.7e-15;         // Magnetic moment for each cell
const double phase_stdev = 1.0;         // St. dev. of intracellular
                                        // phase accumulation

/* Related to the simulation bounds */
const double bound = 40;                // full box is [0, bound]^3 (microns)

/* All water molecules begin the simulation in a box with dimension
   water_start_bound^3 that is centered in the middle of the larger
   simulation box. Given in microns. */
const double water_start_bound = 10;


/* Parameters related to the streamlined nearest cell finder */
const int hashDim = 20;
const int maxNeighbors = 13;

/* Constants affecting diffusion */
const double D_cell = .5547;            // D in micron^2 per ms
const double D_extra = 1.6642;          // D in micron^2 per ms
const double P_expr = 0.01;            // permeability in micron per ms

/* Time scales and step sizes */
const double tau = 1e-6;                // time step in ms
const int totaltime = 1;                // total time to run for in ms - because of GPU architecture, this
                                        // is constrained to be a discrete integer

const int t = (int)(totaltime/tau);     // Total time steps
const double taucp = 5.5;               // Carr-Purcell time in ms
const int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time

#endif /* PARAMETERS_H */
