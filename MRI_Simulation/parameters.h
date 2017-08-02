/*
 * @author  Aadyot Bhatnagar
 * @date    August 17, 2016
 * @file    parameters.h
 * @brief   A file containing static const instantiations of all the paramaters
 *          that affect the way the simulation is conducted.
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

/* Parameters affecting nanoparticle residency in nodes */
#undef FULL_BOUNDARIES      // use full boundary conditions to calculate field?
const double raw_scale = 7; // calculate B explicitly within scale*R of cluster
#ifndef FULL_BOUNDARIES     // otherwise, apply BC's at some cutoff distance
const double border = 6;    // boundary from box where we start applying BC's
#endif

/* Parameters affecting the T2 simulation */
#undef EXPLICIT             // calculate B field explicitly?
const int num_threads = 20;  // number of threads to run T2 simulation on
const int num_runs = 1;     // number of times to run T2 simulation

/* Switches for enabling or disabling debugging output files */
#undef DEBUG_DIFF           // create output file w/ RMS displacements?
#define DEBUG_MNPS          // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?
#undef TIMED_OUTPUT         // print out a progress report every 1ms?


/* Molecule and nanoparticle info */
const int num_water = 2000;              // number of waters in simulation
const double mnp_radius = .1;           // radius of one nanoparticle (um) (.1 NV)

// Exactly ONE of the three flags below must be set
#undef EXTRACELLULAR                    // MNPs intracellular or extracellular?
#define INTRACELLULAR
#undef INTRA_EXTRA

// Exactly ONE of the two flags below must be set
#undef UNCLUSTERED
#define CLUSTERED

/**
 * This flag should only be defined if the flags CLUSTERED and EXTRACELLULAR
 * are also set (guards in place so this occurs). When the flag is set,
 * clustered MNPs are thrown anywhere in extracellular space within the
 * simulation bound for a particular labelled cell.
 */
#if defined CLUSTERED && defined EXTRACELLULAR
#undef THROW_FREE // This flag can be toggled on and off
#endif

const double u_throw_coeff = 1.5;
const double lipid_width = 0.002;        // Lipid width surrounding intracellular MNPs in um
#define LIPID_ENVELOPE

#ifdef UNCLUSTERED
const int num_mnps = (1.760e3);           // number of MNPs (if all extracellular) (1.76e3 NV)
const double mmoment = 1.7e-15;           // magnetic moment of each MNP (1.7e-15 NV)
const double scale = raw_scale;           // to account for smaller MNPs
#elif defined CLUSTERED
const double mnp_pack = 3;              // influences MNP cluster packing
const double scale = raw_scale;         // to account for larger MNPs
#endif

/* Characteristics of FCC cell lattice */
const double cell_r = 9;                // cell radius in microns
const double prob_labeled = 0.26;       // probability a given cell is labeled
const double fcc_pack = 1.00;           // influences FCC packing efficiency
const double bound = 6*sqrt(2)*cell_r*fcc_pack; // full box is [0, bound]^3

/* Constants affecting diffusion */
const double D_cell = .5547;            // D in micron^2 per ms
const double D_extra = 1.6642;          // D in micron^2 per ms
const double P_expr = 0.01;             // permeability in micron per ms
const int hashDim = 50;

/* Time scales and step sizes */
const double tau = 1e-6;                // time step in ms (1e-6 NV)
const double totaltime = 67.0;          // total time to run for in ms
const int t = (int)(totaltime/tau);     // total time steps
const double taucp = 5.5;               // Carr-Purcell time in ms
const int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time


#endif /* PARAMETERS_H */
