/*
 * @author  Aadyot Bhatnagar
 * @date    August 17, 2016
 * @file    parameters.h
 * @brief   A file containing static const instantiations of all the paramaters
 *          that affect the way the simulation is conducted. This file has
 *          been modified to mark certain constants as external variables (to
 *          allow the program to sweep over different values for those
 *          simulation parameters.
 */

#include <cmath>

#ifndef SWEEP_PARAMS_H
#define SWEEP_PARAMS_H

const double mnp_density = 5.18e-9;                 // Density of MNP in mg per micron^3
const double iron_fraction = 0.72;                  // Mass fraction of iron per MNP
const double magnetite_moment = 5e-4;               // MNP Magnetic moment in A . m^2 / mg Iron

const int num_mnp_sizes = 3;                        // Number of nanoparticle sizes to sweep over
const double mnp_sizes[] = {0.03, 0.04, 0.1};       // List of nanoparticle radii (um)

const int num_permeabilities = 3;                   // Number of permeabilities to sweep over
const double permeabilities[] = {0.01, 0.05, 0.20}; // List of permeabilities in micron per ms

const int num_conc = 3;                             // Number of concentrations of MAGNETITE to sweep over
const double concentrations[] = {0.01, 0.1, 1.0};   // List of MNP concentrations in mg / mL


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
#undef DEBUG_MNPS           // create output file w/ all MNP coordinates?
#undef DEBUG_TREE           // check water/node residency via assertions?
#undef DEBUG_FIELD          // create output file w/ B_z at all leaf nodes?
#undef TIMED_OUTPUT         // print out a progress report every 1ms?

/* Molecule and nanoparticle info */
const int num_water = 1000;              // number of waters in simulation
extern double mnp_radius;                // radius of one nanoparticle (um)
#define EXTRACELLULAR                    // MNPs intracellular or extracellular?
#ifdef EXTRACELLULAR
extern int num_mnps;                    // number of MNPs (if all extracellular)
extern double mmoment;                  // magnetic moment of each MNP
const double scale = raw_scale;         // to account for smaller MNPs
#else
const double mnp_pack = 3;              // influences MNP cluster packing
const double scale = raw_scale;         // to account for larger MNPs
#endif

/* Characteristics of FCC cell lattice */
const double cell_r = 2;                // cell radius in microns
const double prob_labeled = 0.25;       // probability a given cell is labeled
const double fcc_pack = 1.00;           // influences FCC packing efficiency
const double bound = 6*sqrt(2)*cell_r*fcc_pack; // full box is [0, bound]^3

/* Constants affecting diffusion */
const double D_cell = .5547;            // D in micron^2 per ms
const double D_extra = 1.6642;          // D in micron^2 per ms
extern double P_expr;                   // permeability in micron per ms

/* Time scales and step sizes */
const double tau = 1e-6;                // time step in ms
const double totaltime = 40.0;          // total time to run for in ms
const int t = (int)(totaltime/tau);     // total time steps
const double taucp = 5.5;               // Carr-Purcell time in ms
const int tcp = (int)(taucp/tau);       // time steps per Carr-Purcell time

#endif /* PARAMETERS_H */
