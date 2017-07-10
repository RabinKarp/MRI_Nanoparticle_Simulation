/*
 * @author  Aadyot Bhatnagar
 * @date    August 17, 2016
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
#ifndef FULL_BOUNDARIES     // otherwise, apply BC's at some cutoff distance
const double border = 6;    // boundary from box where we start applying BC's
#endif

/* Parameters affecting the T2 simulation */
#undef EXPLICIT             // calculate B field explicitly?
const int num_threads = 8;  // number of threads to run T2 simulation on
const int num_runs = 1;     // number of times to run T2 simulation

/* Switches for enabling or disabling debugging output files */
#undef DEBUG_DIFF           // create output file w/ RMS displacements?
#undef DEBUG_MNPS          // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?
#undef TIMED_OUTPUT         // print out a progress report every 1ms?

/* Molecule and nanoparticle info */
const int num_water = 100;              // number of waters in simulation
const double mnp_radius = .1;           // radius of one nanoparticle (um)

// Exactly ONE of the three flags below must be defined.
#undef EXTRACELLULAR                   // MNPs intracellular, extracellular,
#undef INTRACELLULAR                   // or both?
#define INTRA_EXTRA

#define CLUSTERED                      // MNPs clustered or unclustered?
#undef UNCLUSTERED

#define LIPID_ENVELOPE                  // Lipid envelope around intracellular
                                        // MNPs

#ifdef UNCLUSTERED
const double mmoment = 2.0e-15;         // magnetic moment of each MNP
const double scale = raw_scale;         // to account for smaller MNPs
#elif defined CLUSTERED
const double mnp_pack = 3;              // influences MNP cluster packing
const double scale = raw_scale;         // to account for larger MNPs

const double u_throw_coeff = 1.1;       // Clustered extracell. rad coefficient
const double ie_ratio = 0.5;            // When INTRA_EXTRA is set, the ratio
                                        // of intracellular to extracellular MNPs
#endif

const int num_mnps = 1.760e3;           // number of unclustered MNPs
                                        // OR the number of extracellular clustered MNP's

const double lipid_width = 0.002;       // To account for the lipid bilayer
                                        // enveloping intracellular MNP's (um)

/* Characteristics of FCC cell lattice */
const double cell_r = 9;                // cell radius in microns
const double prob_labeled = 0.25;       // probability a given cell is labeled
const double fcc_pack = 1.00;           // influences FCC packing efficiency
const double bound = 6*sqrt(2)*cell_r*fcc_pack; // full box is [0, bound]^3

/* Constants affecting diffusion */
const double D_cell = .5547;            // D in micron^2 per ms
const double D_extra = 1.6642;          // D in micron^2 per ms
const double P_expr = 0.01;             // permeability in micron per ms

/* Time scales and step sizes */
const double tau = 1e-6;                // time step in ms
const double totaltime = 34.0;          // total time to run for in ms
const int t = (int)(totaltime/tau);     // total time steps
const double taucp = 5.5;               // Carr-Purcell time in ms
const int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time

#endif /* PARAMETERS_H */
