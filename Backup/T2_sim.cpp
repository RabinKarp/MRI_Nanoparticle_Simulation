/*
 * @author  Aadyot Bhatnagar
 * @date    July 1, 2016
 * @file    T2_sim.cpp
 * @brief   Includes code for the traversals of the octree and array of water
 *          molecules to apply the phase kicks that cause T2 relaxation.
 */

#include <thread>
#include <iostream>
#include <fstream>
#include <mutex>
#include "octree.h"
#include "fcc_diffusion.h"

#define EXPLICIT 1 // calculate B field explicitly?
#define NUM_THREADS 8
#define PRINTFREQ 1000

const double g = 42.5781e6; // gyromagnetic ratio in MHz/T
extern const int num_water;
extern const int t;
static double mag[NUM_THREADS][t/PRINTFREQ+1];
static mutex mtx;

/*
 * Simulates a step of diffusion for all the water molecules in the FCC lattice
 * and then updates the nodes of the octree where necessary. and applies the
 * phase kick it experiences at its new position.
 */
void diffusion_T2_step(water_info *molecules, Octree *tree, FCC *lattice,\
        int ind_L, int ind_R)
{
    // checked by HD -- correct
    water_info *w = molecules + ind_L, *end = molecules + ind_R;
    for(w; w < end; w++)
    {
        oct_node *voxel = lattice->diffusion_step(w, tree);
#if EXPLICIT
        double B = tree->part_field(w->x, w->y, w->z);
#else
        double B;
        if (voxel->B == 0)
            B = tree->field(w->x, w->y, w->z);
        else
            B = voxel->B;
#endif
        w->phase += B * 2 * pi * g * tau * 1e-3;
    }
}

/*
 * Each thread is responsible for simulating diffusion for (a) some portion of
 * the array of water molecules and (b) some portion of the vector representing
 * physical space.
 */
void thread_function(int thread_no, Octree *tree, water_info *molecules)
{
    // initialize FCC lattice for this thread
    FCC *lattice = new FCC(cell_r, R_io, R_oi, thread_no);

    // initialize indices of iteration for both waters and tree for this thread
    int ind_L = num_water / NUM_THREADS * thread_no;
    int ind_R;
    if (thread_no != NUM_THREADS - 1)
        ind_R = num_water / NUM_THREADS * (thread_no + 1);
    else
        ind_R = num_water;

    defer_lock_t defer;
    unique_lock<mutex> lock(mtx, defer);

    // simulate diffusion for t timesteps
    for (int i = 0; i < t; i++)
    {
        // sum up the phases every PRINTFREQ'th time step and store them
        if (i % PRINTFREQ == 0)
        {
            water_info *w = molecules + ind_L, *end = molecules + ind_R;
            double magnetization = 0;
            for (w; w < end; w++)
                magnetization += cos(w->phase);
            mag[thread_no][i/PRINTFREQ] = magnetization;
        }

        if (i % (2*tcp) == tcp)
        {
            water_info *w = molecules + ind_L, *end = molecules + ind_R;
            for (w; w < end; w++)
                w->phase *= -1;
            lock.lock();
            cerr << "Thread " << thread_no << " applying echo.";
            lock.unlock();
        }

        if (i % (2*tcp) == 0)
        {
            double pct = mag[thread_no][i/PRINTFREQ] / mag[thread_no][0] * 100;
            lock.lock();
            cerr << "Thread " << thread_no << " refocusing. Mag @ " << pct << "% of max." << endl;
            lock.unlock();
        }

        if (i % 100000 == 0)
        {
            double now = (double)(i) * tau;
            lock.lock();
            cerr << "Thread " << thread_no << " simulated " << now << " ms" << endl;
            lock.unlock();
        }

        // simulate one step of diffusion and apply the appropriate phase kicks
        diffusion_T2_step(molecules, tree, lattice, ind_L, ind_R);
    }
    delete lattice;
}

/*
 * Use some number of threads to simulate diffusion through a B field for
 * some number of time steps.
 */
int main(void)
{
    // Initialize octree
#if EXPLICIT
    double max_product = 1e-4, max_g = 40, min_g = 20;
#else
    double max_product = 1e-4, max_g = 1, min_g = 3*sqrt(2)*cell_r/4000;
#endif
    Octree *tree = new Octree(max_product, max_g, min_g, NUM_THREADS, 2);
    cout << "Octree built!" << endl;

    // Initialize PRNG and use it to seed water molecules
    XORShift<> gen(time(NULL));
    for (int i = 0; i <= NUM_THREADS; i++)
        gen.jump();
    water_info *w = init_molecules(sqrt(2)*cell_r, num_water, tree->mnps, gen);
    
    // Simulate T2 relaxation using the number of threads specified
    vector<thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(thread_function, i, tree, w); 
    for (vector<thread>::iterator i = threads.begin(); i < threads.end(); i++)
        i->join();

    // Clean up octree & array of water molecules
    delete tree;
    delete[] w;

    // Print the output from each thread into an output file
    ofstream out_file;
    string filename("T2_sim_");
    filename += to_string((int)(tau * 1e9));
    filename += "ps_";
    filename += to_string(num_water);
    filename += "w";
#if EXPLICIT
    filename += "_ex";
#endif
    filename += ".csv";
    out_file.open(filename);
    for (int i = 0; i < t / PRINTFREQ; i++)
    {
        out_file << i * PRINTFREQ * tau << ",";
        double magnetization = 0;
        for (int j = 0; j < NUM_THREADS; j++)
        {
            out_file << mag[j][i] << ",";
            magnetization += mag[j][i];
        }
        out_file << magnetization << endl;
    }
    out_file.close();
}
