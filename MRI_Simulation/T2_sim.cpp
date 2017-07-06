/*
 * @author  Aadyot Bhatnagar
 * @date    July 27, 2016
 * @file    T2_sim.cpp
 * @brief   Includes code for the traversals of the octree and array of water
 *          molecules to apply the phase kicks that cause T2 relaxation.
 */

#include <thread>
#include <iostream>
#include <mutex>
#include <cassert>
#include "octree.h"
#include "fcc_diffusion.h"
#include "parameters.h"

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T
const int pfreq = (int)(1e-3/tau);      // print net magnetization every 1us
static double **mag;                    // net magnetization data
static std::mutex mtx;                  // regulates access to print buffers

#ifdef DEBUG_DIFF
const int dfreq = (int)(1e-3/tau);      // print diffusion info every 1us
static double **disp;                   // absolute values of displacements
static double *init_pos;                // initial positions
#endif

/*
 * Simulates a step of diffusion for all the water molecules in the FCC lattice
 * and then updates the nodes of the octree where necessary. and applies the
 * phase kick it experiences at its new position.
 */
void diffusion_T2_step(water_info *molecules, Octree *tree, FCC *lattice,\
        int ind_L, int ind_R, XORShift<> &gen)
{
    water_info *end = molecules + ind_R;
    for(water_info *w = molecules + ind_L; w < end; w++)
    {
        double B = lattice->diffusion_step(w, tree, gen);
        w->phase += B * 2 * pi * g * tau * 1e-3;
    }
}

/*
 * Each thread is responsible for simulating diffusion for (a) some portion of
 * the array of water molecules and (b) some portion of the vector representing
 * physical space.
 */
void thread_func(const int tid, Octree *tree, water_info *molec, FCC *lattice)
{
    // initialize PRNG for this thread
    std::random_device rd;
    XORShift<> gen(time(NULL) + rd());
    for (int i = 0; i < tid; i++)
        gen.jump();

    // initialize indices of iteration for both waters and tree for this thread
    int ind_L = num_water / num_threads * (tid % num_threads);
    int ind_R;
    if (tid != num_threads - 1)
        ind_R = num_water / num_threads * (tid % num_threads + 1);
    else
        ind_R = num_water;

    std::defer_lock_t defer;
    std::unique_lock<std::mutex> lock(mtx, defer);

    // simulate diffusion for t timesteps
    for (int i = 0; i < t; i++)
    {
        // sum up the phases every microsecond and store them
        if (i % pfreq == 0)
        {
            water_info *end = molec + ind_R;
            double magnetization = 0;
            for (water_info *w = molec + ind_L; w < end; w++)
                magnetization += cos(w->phase);
            mag[tid][i/pfreq] = magnetization;
        }

#ifdef DEBUG_DIFF
        // if debugging diffusion, store mean diffusion length in each direction
        if (i % dfreq == 0)
        {
            double *init = init_pos + 3*ind_L;
            water_info *end = molec + ind_R;
            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (water_info *w = molec + ind_L; w < end; w++)
            {
                sum_x += abs(w->x - init[0] + 2*bound*(double)w->cross_x);
                sum_y += abs(w->y - init[1] + 2*bound*(double)w->cross_y);
                sum_z += abs(w->z - init[2] + 2*bound*(double)w->cross_z);
                init += 3;
            }
            disp[tid][3*i/dfreq+0] += sum_x / num_water / num_runs;
            disp[tid][3*i/dfreq+1] += sum_y / num_water / num_runs;
            disp[tid][3*i/dfreq+2] += sum_z / num_water / num_runs;
        }
#endif /* DEBUG_DIFF */
        
#ifdef TIMED_OUTPUT
        // print out a progress report every million time-steps 
        if (i % 1000000 == 0)
        {
            double now = (double)(i) * tau;
            lock.lock();
            std::cerr << "Thread " << tid << " simulated " << now << " ms.";
            std::cerr << std::endl;
            lock.unlock();
        }
#endif /* TIMED_OUTPUT */
        
        // apply spin echo and flip phases every odd Carr-Purcell time
        if (i % (2*tcp) == tcp)
        {
            water_info *end = molec + ind_R;
            for (water_info *w = molec + ind_L; w < end; w++)
                w->phase *= -1;
            lock.lock();
            std::cerr << "Thread " << tid << " applying echo at ";
            std::cerr << (double)i*tau << "ms." << std::endl;
            lock.unlock();
        }

        // output a message with the magnetization value every echo time
        if (i % (2*tcp) == 0)
        {
            double pct = mag[tid][i/pfreq] / mag[tid][0] * 100;
            lock.lock();
            std::cerr << "Thread " << tid << " refocusing at " << (double)i*tau;
            std::cerr << "ms. Magnetization at " << pct << "% of max.";
            std::cerr << std::endl;
            lock.unlock();
        }

        // simulate one step of diffusion and apply the appropriate phase kicks
        diffusion_T2_step(molec, tree, lattice, ind_L, ind_R, gen);
    }
}

/*
 * Generates a detailed filename for an output file, containing information
 * about the parameters used to generate it.
 */
std::string generate_base_filename()
{
    std::string filename("tau=");
    filename += std::to_string((unsigned)(tau * 1e9));
    filename += "ps_T-e=";
    filename += std::to_string((unsigned)(2*taucp));
    filename += "ms_pct-labeled=";
    filename += std::to_string((unsigned)(100*prob_labeled));

#ifdef EXTRACELLULAR
    filename += "_unclustered";
#else
    filename += "_clustered";
    if (mnp_pack < 2.99)
        filename += "_tight-mnps";
#endif
    if (fcc_pack > 1.01)
        filename += "_loose-cells";

#ifdef EXPLICIT
    filename += "_ex_";
#endif

#ifndef FULL_BOUNDARIES
    filename += "_border=";
    filename += std::to_string((unsigned)(border));
    filename += "um_";
#endif

    filename += std::to_string((unsigned)(time(NULL)));
    return filename;
}

/*
 * Use some number of threads (specified at the top of this file) to simulate
 * the diffusion of some number of water molecules (specified at the top of the
 * file fcc_diffusion.h) through a B field for some number of time steps (also
 * specified in fcc_diffusion.h).
 */
int main(void)
{
    // Initialize array containing magnetization values at each time step
    mag = new double*[num_threads];
    for (int i = 0; i < num_threads; i++)
        mag[i] = new double[t/pfreq];

    // temporary output filenames
    std::string *temp_filenames = new std::string[num_runs];
    std::ifstream *temp_files = new std::ifstream[num_runs];

#ifdef DEBUG_DIFF
    // initialize arrays containing initial positions and square displacements
    init_pos = new double[3*num_water];
    disp = new double*[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        disp[i] = new double[3*t/dfreq];
        for (int j = 0; j < 3*t/dfreq; j++)
            disp[i][j] = 0;
    }
#endif /* DEBUG_DIFF */

    // run the full simulation the specified number of times
    for (int i = 0; i < num_runs; i++)
    {
        // Initialize PRNG and use it to seed the nanoparticles & waters
        std::random_device rd;
        XORShift<> gen(time(NULL) + rd());

        // Initialize FCC lattice and octree
        FCC *lattice = new FCC(D_cell, D_extra, P_expr);
        std::vector<MNP_info> *mnps = lattice->init_mnps(gen);
        double max_product = 2e-6, max_g = 5, min_g = .002;
        uint64_t start = time(NULL);
        Octree *tree = new Octree(max_product, max_g, min_g, gen, mnps);
        uint64_t elapsed = time(NULL) - start;
        std::cout << "Octree took " << elapsed / 60 << ":";
        if (elapsed % 60 < 10) std::cout << "0";
        std::cout << elapsed % 60 << " to build." << std::endl << std::endl;
    #ifdef DEBUG_FIELD
        std::cout << "Since field debugging does not support any explicit ";
        std::cout << "calculation, program is exiting." << std::endl;
        exit(0);
    #endif

        // Initialize water molecules
        for (int j = 0; j <= num_threads * (i+1); j++)
            gen.jump();
        water_info *w = lattice->init_molecules(bound, num_water, mnps, gen);

    #ifdef DEBUG_DIFF
        // initialize starting positions for all molecules
        for (int j = 0; j < num_water; j++)
        {
            init_pos[3*j+0] = w[j].x;
            init_pos[3*j+1] = w[j].y;
            init_pos[3*j+2] = w[j].z;
        }
    #endif /* DEBUG_DIFF */
        
        // Simulate T2 relaxation using the number of threads specified
        start = time(NULL);
        std::vector<std::thread> thds;
        for (int j = 0; j < num_threads; j++)
            thds.emplace_back(thread_func, j, tree, w, lattice); 
        for (auto &t : thds)
            t.join();
        elapsed = time(NULL) - start;
        std::cout << "Simulation (w/o tree) took " << elapsed / 60 << ":";
        if (elapsed % 60 < 10) std::cout << "0";
        std::cout << elapsed % 60 << " to run." << std::endl << std::endl;

        // Clean up dynamically allocated resources
        delete[] w;
        delete tree;
        delete lattice;
        delete mnps;

        // Print output to a temporary file
        temp_filenames[i] = generate_base_filename();
        temp_filenames[i] += ".csv";
        std::ofstream out_file;
        out_file.open(temp_filenames[i]);
        for (int j = 0; j < t/pfreq; j++)
        {
            double magnetization = 0;
            for (int k = 0; k < num_threads; k++)
                magnetization += mag[k][j];
            out_file << magnetization << std::endl;
        }
        out_file.close();
    }

    // Initialize each run's output as an ifstream & create a final output file
    std::ofstream final_out;
    std::string filename = generate_base_filename();
    filename += "_finalhighdiffusion.csv";
    final_out.open(filename);
    for (int i = 0; i < num_runs; i++)
        temp_files[i].open(temp_filenames[i]);

    // Sum up the output from each run into the final output file
    for (int i = 0; i < t/pfreq; i++)
    {
        final_out << i * pfreq * tau;
        double net_mag = 0;
        for (int j = 0; j < num_runs; j++)
        {
            double mag = 0;
            temp_files[j] >> mag;
            final_out << "," << mag;
            net_mag += mag;
        }
        final_out << "," << net_mag << std::endl;
    }

    // Close all files and delete temporary files
    final_out.close();
    for (int i = 0; i < num_runs; i++)
    {
        temp_files[i].close();
        remove(temp_filenames[i].data());
    }
    delete[] temp_filenames;
    delete[] temp_files;

#ifdef DEBUG_DIFF
    /* If debugging diffusion, output a file containing the absolute values of
     * the mean of the absolute values of the displacements of all molecules in 
     * the x, y, and z directions. */
    std::ofstream out_file;
    out_file.open("T2_sim_diffusion_stats.csv");
    out_file << "t,x,y,z" << std::endl;
    for (int i = 0; i < t / dfreq; i++)
    {
        out_file << i * dfreq * tau << ",";
        double sum_x = 0, sum_y = 0, sum_z = 0;
        for (int j = 0; j < num_threads; j++)
        {
            sum_x += disp[j][3*i+0];
            sum_y += disp[j][3*i+1];
            sum_z += disp[j][3*i+2];
        }
        out_file << sum_x << "," << sum_y << "," << sum_z << std::endl;
    }
    out_file.close();
#endif /* DEBUG_DIFF */
}
