/*
 * @author  Vivek Bharadwaj
 * @date    May 22, 2016
 * @file    p_sweep_sim.cpp
 * @brief   Simple piece of code that sweeps the T2 contrast simulation over
 *          a range of parameters.
 */

#include "sweep_parameters.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include "octree.h"
#include "fcc_diffusion.h"
#include "T2_sim.cpp"

using namespace std;

#define PROGRESS_REPORT // Switch that prints out a progress report every time
                        // a parameter changes

int num_mnps;
double P_expr;                // permeability in micron per ms
double mnp_radius;            // radius of one nanoparticle (um)
double mmoment;               // magnetic moment of each MNP

const string l_delim = ",";
const string e_delim = ",,";

/**
 * Returns the volume of a sphere with the specified radius.
 *
 * @param  Radius of the sphere
 * @return Volume of the sphere
 */
double sphere_volume(double radius) {
  return (4.0 / 3.0) * M_PI * pow(radius, 3);
}

/**
 * Returns the magnetic moment for a nanoparticle given its radius. Computes the
 * volume of a nanoparticle, which it uses to get the mass of the MNP (based
 * on the density of magnetite). The mass is scaled by the iron fraction in
 * magnetite and multiplied by the magnetic moment of the particle.
 *
 * @param radius MNP radius in microns
 *
 * @return Magnetic moment in units of A . m^2
 */
double magneticMoment(double radius) {
	double volume = sphere_volume(radius); // Units of microns^3
	return volume * mnp_density * iron_fraction * magnetite_moment; // Units A.m^2
}

/**
 * Returns the number of nanoparticles of a specified radius required within the
 * simulation volume to maintain the iron concentration at a specified value.
 * This function relies on the density of a magnetic nanoparticle defined in
 * the parameter file.
 *
 * @param concentration The target magnetite concentration, given in mg / mL
 * @param radius        The radius of each magnetic nanoparticle, given in
 *                      microns
 *
 * @param The target number of magnetic nanoparticles of the specified radius
 *        to achieve the target concentration
 */

int getMNPCount(double concentration, double radius) {
	// Figure out the total amount of magnetite required in the volume
	double box_volume = pow(bound, 3); // Units of microns^3
	double targetAmount = box_volume * concentration * pow(10, -12); // Units of mg

	// Figure out the amount of magnetite in a single MNP
	double MNP_amount = sphere_volume(radius) * mnp_density; // Units of mg

	// Divide and return the total number of MNP's
	return (int) (targetAmount / MNP_amount);
}

/**
 * Write the output list to a file - the output is of the form
 * psweep_[RANDOM_NUMBER].csv
 */
void generateSweepOutput(string fileList[][num_permeabilities][num_conc]) {
	cout << "Writing output file" << endl;
	// Generate the filename
	string fName = "psweep_";
	fName += std::to_string((unsigned)(time(NULL)));
	fName += ".csv";
	ofstream fout(fName);

	// Generate the output file header
	fout << "MNP Size (um)" << l_delim
		<< "Permeability (um / ms)" << l_delim
		<< "Concentration (mg / mL)" << l_delim
		<< "Filename" << e_delim
		<< endl;

	for(int i = 0; i < num_mnp_sizes; i++) {
		for(int j = 0; j < num_permeabilities; j++) {
			for(int k = 0; k < num_conc; k++) {
				fout << mnp_sizes[i] << l_delim
					<< permeabilities[j] << l_delim
					<< concentrations[k] << l_delim
					<< fileList[i][j][k] << e_delim
					<< endl;
			}
		}
	}
	fout.close();
}

/**
 * Main method that sweeps over three parameters (radii of MNP's, permeabilities
 * of cell boundaries, and concentrations of magnetite and runs the T2 simulation
 * for each set of paraemters. The result of each simulation is written to a file;
 * the complete list of filenames outputted by the program has the filename
 * p_sweep_[TIME].csv, where [TIME] uniquely stamps the file to avoid name conflicts.
 */
int main(int argc, char** argv) {
	cout << "Starting Parameter Sweep" << endl;

	string results[num_mnp_sizes][num_permeabilities][num_conc];

	// Sweep over each of the three parameters
	for(int i = 0; i < num_mnp_sizes; i++) {
    	for(int j = 0; j < num_permeabilities; j++) {
			     for(int k = 0; k < num_conc; k++) {

		        // Initialize all simulation parameter variables from the arrays
        		mnp_radius = mnp_sizes[i];
            P_expr = permeabilities[j];
            num_mnps = getMNPCount(concentrations[k], mnp_radius);
            mmoment = magneticMoment(mnp_radius);

#ifdef PROGRESS_REPORT
            cout << "Simulating radius " << mnp_radius
				        << " microns, concentration " << concentrations[k]
				        << " mg / mL, permeability " << P_expr
			          << " microns per millisecond." << endl;

			      cout << "Computed Magnetic Moment per MNP: "
              << mmoment << " A . m^2" << endl;
#endif

				// Run T2 simulation
				results[i][j][k] = simulateWaterMolecules();
			}
		}
	}
	cout << "Generating output..." << endl;
	cout << results[0][0][0] << endl;
	generateSweepOutput(results);
}
