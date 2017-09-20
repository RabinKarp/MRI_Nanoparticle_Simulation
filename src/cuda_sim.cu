/**
 * @author  Vivek Bharadwaj
 * @date    17 September 2017.
 * @file    cuda_sim.cu 
 * @brief   Contains kernels for GPU simulation of diffusing water molecules. 
 *
 * This file contains functions to execute the GPU simulation of diffusing
 * water molecules. It relies on the Simulation Box class to initialize
 * the starting positions of the waters, the magnetic nanoparticles,
 * the cells that form diffusion boundaries, and the octree for fast
 * field computation.
 *
 * Because CUDA programming idioms are C-like, none of the functions below
 * have been encapsulated within classes.
 */

#include "utilities/utilities.h"
#include "cuda_sim.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include "random"

#include <vector>
#include <fstream>
#include <thread>
#include <iostream>
#include "math.h"

#include "parameters.h"
#include "BacteriaBox.h"
#include "gpu_random.h"
#include "octree.h"

using namespace std;

const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T
const double pInt = 1e-3;
const int pfreq = (int)(pInt/tau);      // print net magnetization every 1us

// Constants for lin. alg. operations
const double alpha = 1;
const double beta = 0;

#define num_uniform_doubles 5 // # of uniform doubles per water per tstep 
#define num_normal_doubles 1  // # of normal  doubles per water per tstep

/**
 * The t_rands structure, meant as a register variable for threads in the kernel
 * simulateWaters, stores pointers for the random numbers required by a thread
 * on a single time step of the simulation - three uniform random numbers and
 * a normal random number to generate a random displacement with a given
 * standard deviation, and a uniform random number used as a coin to flip
 * to decide whether or not to diffuse into a cell.
 */
struct t_rands {
    double *x;
    double *y;
    double *z;
    double *coin;
    double *norm;

    /**
     * Increments all of the pointers encapsulated by this struct by the
     * same given integer.
     *
     * @param stride    The integer amount to increment the pointers
     */
    __device__ void advancePointer(int stride) {
        x += stride;
        y += stride;
        z += stride;
        coin += stride;
        norm += stride;
    }
};

//==============================================================================
// Octree-related functions
//==============================================================================

/**
 * GPU function that computes and returns the field produced by a magnetic 
 * dipole with moment M at a point specified by <dx, dy, dz> relative to the 
 * dipole itself. As currently implemented, the function will return 0 if
 * the norm of the vector <dx, dy, dz> is less than the cell radius,
 * since we're using a different mechanism to handle phase kicks within cells. 
 *
 * @param   dx The x-displacement relative to the dipole
 * @param   dy The y-displacement relative to the dipole
 * @param   dz The z-displacement relative to the dipole
 * @param   M  The magnetic moment of the dipole
 * @param   d  A reference to a GPUData object containing relevant parameters
 *
 * @return  The magnetic field, in Teslas, at the location specified by the
 *          input position vector
 */
__device__ double dipole_field(double dx, double dy, double dz, 
    double M, GPUData &d) {
    
    double sqDist = NORMSQ(dx, dy, dz);
    double divisor = sqDist * sqDist * sqrt(sqDist);
    return (sqDist > d.cell_r * d.cell_r) * M * 1e11 
        * (2*dz*dz - dx*dx - dy*dy) / divisor;
}

/**
 * GPU function that computes and returns the morton code at a specified depth
 * for a specified (x, y, z) point by interleaving bits. This function is the
 * GPU analogue of the same function on the CPU, defined within the octree class.
 *
 * @param depth The depth of the morton code to return
 * @param x     The x-coordinate to get the morton code for
 * @param y     The y-coordinate to get the morton code for
 * @param z     The z-coordinate to get the morton code for
 * @param d     A reference to a GPUData object containing relevant parameters
 *
 * @return A morton code for the (x, y, z) point at the specified depth 
 */
inline __device__ uint64_t morton_code(int depth, double &x, double &y, 
    double &z, GPUData &d) {
   
    uint64_t size = 1 << (depth);
    uint32_t idx_x = floor(x / d.bound * size);
    uint32_t idx_y = floor(y / d.bound * size);
    uint32_t idx_z = floor(z / d.bound * size);
    uint64_t answer = 0;
    // start by shifting the third byte, since we only look @ the first 21 bits
    if (depth > 16)
    {
        answer |=   d.morton_z[(idx_z >> 16) & 0xFF ] |
                    d.morton_y[(idx_y >> 16) & 0xFF ] |
                    d.morton_x[(idx_x >> 16) & 0xFF ];
        answer <<= 24;
    }

    // shift second byte
    if (depth > 8)
    {
        answer |=   d.morton_z[(idx_z >> 8) & 0xFF ] |
                    d.morton_y[(idx_y >> 8) & 0xFF ] |
                    d.morton_x[(idx_x >> 8) & 0xFF ];
        answer <<= 24;
    }

    // shift first byte
    answer |=   d.morton_z[(idx_z) & 0xFF ] |
                d.morton_y[(idx_y) & 0xFF ] |
                d.morton_x[(idx_x) & 0xFF ];

    return answer;
}

/**
 * Finds the subtree of the octree on the GPU that contains the given
 * (wx, wy, wz) coordinate. This is the analogue of the find_tree function
 * defined in the Octree class.
 *
 * @param wx    The x-coordinate to get the subtree for
 * @param wy    The y-coordinate to get the subtree for
 * @param wz    The z-coordinate to get the subtree for
 * @param d     A reference to a GPUData object containing relevant parameters
 *
 * @return A pointer to a pointer to a GPU node, representing a pointer to
 *         the subtree of the octree containing the given (wx, wy, wz)
 *         coordinates
 */
inline __device__ gpu_node** find_tree(double wx, double wy, double wz, 
        GPUData &d)
{
    return d.tree + morton_code(d.min_depth, wx, wy, wz, d);
}

/**
 * GPU helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 *
 * @param wx    The x-coordinate to return the morton code for
 * @param wy    The y-coordinate to return the morton code for
 * @param wz    The z-coordinate to return the morton code for
 * @param d     The depth of the parent node
 * @param data  A reference to a GPUData object containing relevant parameters
 *
 * @return A morton code for the (x, y, z) point at one additional level of 
 *         depth 
 */
inline __device__ unsigned find_child(double wx, double wy, double wz, 
    int d, GPUData &data) {

    return morton_code(d + 1, wx, wy, wz, data) & 7;
}

/**
 * GPU helper function that returns the node of the octree within a particular
 * subtree that contains the given (wx, wy, wz) coordinates. This function
 * navigates the octree recursively, checking if the current node is a leaf
 * and then navigating to the child of that node containing the given water
 * molecule position until the leaf is found. Analogue of the same CPU function
 * defined in the octree class.
 *
 * @param gpu_node  The subtree of the octree containing the water molecule
 * @param wx        The x-coordinate to find the leaf of the octree for
 * @param wy        The y-coordinate to find the leaf of the octree for
 * @param wz        The z-coordinate to find the leaf of the octree for
 * @param depth     To the client function, the depth parameter must be 
 *                  the minimum depth of the octree (i.e. the depth of
 *                  the provided subtree node)
 * @param data      A reference to a GPUData object containing relevant 
 *                  parameters
 * 
 * @return The leaf of the octree that contains the given coordinates
 */
__device__ gpu_node* find_node(gpu_node *n, double wx, double wy, 
        double wz, int d, GPUData &data) {
    
    // base case -- node is a leaf
    if (n->mc >> 63)
        return n;

    // otherwise, navigate to the appropriate child and recurse
    unsigned child_no = find_child(wx, wy, wz, d, data);
    return find_node(n + n->child[child_no].idx, wx, wy, wz, d + 1, data);
}


/**
 * Returns a leaf of the octree containing the water molecule at the provided
 * (wx, wy, wz) coordinates.
 *
 * @param wx    The x-position to get the leaf for
 * @param wy    The y-position to get the leaf for
 * @param wz    The z-position to get the leaf for
 * @param data      A reference to a GPUData object containing relevant 
 *                  parameters
 *
 * @return A pointer to the leaf of the octree containing the given coordinates
 */
__device__ gpu_node* get_voxel(double &wx, double &wy, double &wz, GPUData &d) {
    return find_node(*(find_tree(wx, wy, wz, d)), wx, wy, wz, d.min_depth, d);
}

/**
 * GPU function that returns the B field at the location of a particular 
 * water molecule. The GPU function accomplishes this by extracting the B 
 * field at the provided octree leaf and adding it to the field produced 
 * by all MNPs that are within a fixed radius of that MNP (zeroed out during 
 * octree construction).
 *
 * @param wx    The x-coordinate of the water molecule
 * @param wy    The y-coordinate of the water molecule
 * @param wz    The z-coordinate of the water molecule
 * @param leaf  A pointer to the gpu_node leaf struct where this water resides
 *              in the octree
 * @param d     A reference to a GPUData object containing relevant parameters
 *
 * @return      The B-field, in Teslas, at the parameter location
 */
inline __device__ double get_field(double &wx, double &wy, double &wz, 
        gpu_node* leaf, GPUData &d) {
   
    uint64_t depth = 0, mc = (leaf->mc << 1) >> 1;
    while (mc >>= 3) depth++;

    // use Morton code's depth to find child index to find value of B to return
    unsigned child_no = find_child(wx, wy, wz, depth, d);
    double B = (double)leaf->child[child_no].B;

    // add in contributions from resident MNPs zeroed out during construction
    for(int i = 0; i < leaf->numResidents; i++) {
        MNP_info *np = leaf->resident + i;
        B += dipole_field(wx - np->x, wy - np->y, wz - np->z, np->M, d);
    }

    return B;
}

/**
 * Initialize a GPU verison of the octree from the CPU version. The function
 * handles this through a complex nested copy application. The resulting
 * GPU octree is stored within the GPUData class passed as a reference.
 *
 * @param oct    A pointer to an octree
 * @param d      A reference to a GPUData object containing relevant parameters 
 */
void initOctree(Octree *oct, GPUData &d) {
    // Initialize octree parameters
    d.min_depth = oct->min_depth;
    d.max_depth = oct->max_depth;
    d.addresses = new std::vector<void*>();

    vector<oct_node> *vec_nodes = oct->space;
    vector<MNP_info> &vec_mnps = *(oct->mnps);

    int arr_size = (int) pow(8, d.min_depth);
    d.arr_size = arr_size;

    gpu_node** localTree = new gpu_node*[arr_size];
    gpu_node** localPointers = new gpu_node*[arr_size];
    d.sizes = new int[arr_size];

    bool* checked = new bool[vec_mnps.size()];

    for(int i = 0; i < arr_size; i++) {
        if(vec_nodes + i) {
          vector<oct_node> &current = vec_nodes[i];
          localTree[i] = new gpu_node[current.size()];

          for(int j = 0; j < current.size(); j++) {
              localTree[i][j].mc = current[j].mc;

              for(int k = 0; k < 8; k++) {
                  localTree[i][j].child[k] = current[j].child[k];
              }

              if(current[j].resident) {
                  localTree[i][j].numResidents = current[j].resident->size();

                  // This will become a device pointer
                  localTree[i][j].resident = (MNP_info*) cudaAllocate(sizeof(MNP_info) * localTree[i][j].numResidents);
                  d.addresses->push_back((void*) localTree[i][j].resident);
                  // Copy MNPs to device
                  copyToDevice((void *) localTree[i][j].resident,
                      (void*) current[j].resident->data(),
                      sizeof(MNP_info) * localTree[i][j].numResidents);
              }
              else {
                  localTree[i][j].numResidents = 0;
                  localTree[i][j].resident = nullptr;
              }
          }
          // Store the subtree size locally
          d.sizes[i] = current.size();
          // Now copy the entire subtree to the device, storing in the local pointers struct
          localPointers[i] = (gpu_node *) cudaAllocate(sizeof(gpu_node) * current.size());
          d.addresses->push_back((void*) localPointers[i]);
          copyToDevice((void*)localPointers[i], (void*) localTree[i], sizeof(gpu_node) * current.size());
        }

        // Now copy the entire tree into memory
        d.tree = (gpu_node**) cudaAllocate(sizeof(gpu_node**) * arr_size);
        copyToDevice((void*) d.tree, (void*) localPointers, sizeof(gpu_node**) * arr_size);
    }

    d.morton_x = (uint32_t*) cudaAllocate(256 * sizeof(uint32_t));
    d.morton_y = (uint32_t*) cudaAllocate(256 * sizeof(uint32_t));
    d.morton_z = (uint32_t*) cudaAllocate(256 * sizeof(uint32_t));

    copyToDevice((void*) d.morton_x,(void*) morton_x, 256 * sizeof(uint32_t));
    copyToDevice((void*) d.morton_y,(void*) morton_y, 256 * sizeof(uint32_t));
    copyToDevice((void*) d.morton_z,(void*) morton_z, 256 * sizeof(uint32_t));

    for(int i = 0; i < arr_size; i++) {
        delete[] localTree[i];
    }

    cout << "Allocated GPU Octree!" << endl;
}

/**
 * Destroys the octree stored within a GPUData class, freeing up those resources
 * in the GPU.
 *
 * @param d    A reference to the GPUData class containing the octree
 *             to be destroyed
 */
void destroyTree(GPUData &d) {
    // TODO: Fix memory cleanup here

    cudaFree(d.tree);

    for(std::vector<void*>::iterator it = d.addresses->begin(); it != d.addresses->end(); it++) {
        cudaFree(*it);
    }

    cout << "Tree freed!" << endl;

    // Free the morton code arrays
    cudaFree(d.morton_x);
    cudaFree(d.morton_y);
    cudaFree(d.morton_z);

    // Host pointer, can call delete
    delete[] d.sizes;
    delete d.addresses;
}

//==============================================================================
// GPU setup functions
//==============================================================================

/**
 * Frees the nearest cell lookup table stored on the GPU.
 *
 * @param d     A reference to the GPUData class storing a pointer to the 
 *              nearest cell lookup table on the GPU
 */
void destroyLookupDevice(GPUData &d) {
    for(int i = 0; i < hashDim * hashDim * hashDim; i++) {
        cudaFree(d.localLookup[i]);
    }
    cudaFree(d.lookupTable);
    delete[] d.localLookup;
}

/**
 * Releases the octree and the nearest cell lookup table in GPU memory.
 *
 * @param d     A reference to the GPUData object containing the octree and
 *              nearest cell lookup table to free from memory
 */
void finalizeGPU(GPUData &d) {
    destroyLookupDevice(d); 
    destroyTree(d);
}

/**
 * Copies over parameters defined in CPU memory to variables within the GPUData
 * class passed as a reference, which will be used for constant reference
 * on the GPU.
 *
 * @param d     A reference to the GPUData object to set relevant parameters in
 */
void setParameters(GPUData &d) {
    d.in_stdev = sqrt(M_PI * D_cell * tau);
    d.out_stdev = sqrt(M_PI * D_extra * tau);

    d.reflectIO = reflectIO; 
    d.reflectOI = reflectOI; 
    d.tcp = tcp;

    d.num_cells = num_cells;
    d.num_waters = num_water;
    d.timesteps = sprintSteps;
    d.cell_r = cell_r;
    d.bound = bound;
    d.nBlocks = num_blocks;
    d.g = g;
    d.tau = tau;
    d.bound = bound;
    d.pfreq = pfreq;
    d.hashDim = hashDim;

#ifdef RANDOM_KICK
    d.phase_stdev = phase_stdev;
    d.phase_k = phase_k;
#elif defined CONSTANT_KICK
    d.phase_k = phase_k;
#endif 

}

//==============================================================================
// Water Molecule Simulation functions
//==============================================================================

/**
 * GPU helper function that updates the cell closest to a given water molecule
 * by using the nearest cell lookup table initialized by SimulationBox. The
 * lattice point corresponding to a water molecule is determined, and the
 * function cycles through cells within a fixed distance of the lattice point
 * looking for candidate cells that the parameter water molecule could be
 * inside of.
 *
 * Postcondition: The closest cell record of the provided water molecule
 * is changed to the new cell closest to the water molecule
 *
 * @param w     Pointer to the water to update the nearest cell for
 * @param d     A GPUData class containing relevant parameters
 */
inline __device__ void updateNearest(water_info *w, GPUData &d) {
    double cubeLength = d.bound / hashDim;
    int x_idx = w->x / cubeLength;
    int y_idx = w->y / cubeLength;
    int z_idx = w->z / cubeLength;

    // Get an array of candidate cells close to the lattice point
    int* nearest =
        d.lookupTable[z_idx * d.hashDim * d.hashDim
        + y_idx * d.hashDim
        + x_idx];

    // Some ridiculously high upper bound on the nearest distance
    double cDist = d.bound * d.bound * 3;
    int cIndex = -1;
   
    // Cycle through candidates, determine water molecule residency 
    while(*nearest != -1) {
        double dx = d.lattice[*nearest].x - w->x;
        double dy = d.lattice[*nearest].y - w->y;
        double dz = d.lattice[*nearest].z - w->z;

        double dist = NORMSQ(dx, dy, dz);
        if(NORMSQ(dx, dy, dz) < cDist) {
            cDist = dist;
            cIndex = *nearest;
        }
        nearest++;
    }

    w->in_cell = (cDist < d.cell_r * d.cell_r);
    w->nearest = cIndex;
}

/**
 * Returns true if a water molecule crosses the boundary of the cell closest to
 * it and reflects off the boundary instead of crossing it. Whether or not
 * the reflection occurs depends on a random coin and the permeability
 * parameters of the cells.
 *
 * @param i      The initial position of the water molecule before a given tstep
 * @param f      The final position of the water molecule after a given tstep
 * @param r_nums A structure containing the random coin to determine reflection
 * @param d      A GPUData class containing relevant parameters
 *
 * @return true  If the water molecule both crosses a cell membrane and is
 *               reflected across it,
 *         false Otherwise
 */
inline __device__ bool cell_reflect(water_info *i, water_info *f, 
        t_rands *r_nums, GPUData &d) { 
    
    double coin = *(r_nums->coin); 
    bool flip = (i->in_cell && (! f->in_cell) && coin < d.reflectIO)
                    || ((! i->in_cell) && f->in_cell && coin < d.reflectOI);
    return flip;
}

/**
 * Returns true if a water molecule bumps into an MNP within the provided
 * list and needs to be reflected back. The function relies on the
 * client to supply a list of candidate MNPs that the water molecule could
 * bump into; if the water molecule bumps into an MNP not specified
 * in the candidate list, this funciton will NOT record a reflection.
 *
 * @param w         The water molecule to check for reflection
 * @param mnp       A pointer to an array of MNP candidates to check for
 *                  bumping into
 * @param num_mnps  The number of candidate MNPs in the given array
 * @param d         A GPUData class containing relevant parameters
 *
 * @return true  If the water molecule bumps into one of the candidate MNPs
 *               and reflects off it,
 *         false Otherwise
 */
inline __device__ bool mnp_reflect(water_info *w, MNP_info *mnp, 
        int num_mnps, GPUData &d) {
    
    bool retValue = false;

    for(int i = 0; i < num_mnps; i++) {
        MNP_info* m = mnp + i;
        double dx = m->x - w->x;
        double dy = m->y - w->y;
        double dz = m->z - w->z;

        if(NORMSQ(dx, dy, dz) < (m->r * m->r))
            retValue = true;
    }

    return retValue;
}

/**
 * Returns a normal random displacement for a provided water molecule in a 
 * random direction, with the st. dev of this displacement affected by whether
 * or not a water molecule is inside a cell. The diffusion constants
 * inside and outside the cell affect the standard deviation of the normal
 * distribution of diffusion step sizes. See the formulas within the function
 * for details.
 *
 * @param w         The water molecule to get a random displacement for
 * @param r_nums    A structure containing random numbers to compute
 *                  the random direction and step size from.
 * @param d         A GPUData class containing relevant parameters
 *
 * @return A random displacement, in the form of a water_info water struct 
 *         with (x, y, z) coordinates set to the normal random displacement.
 */
inline __device__ water_info rand_displacement(water_info *w, 
        t_rands *r_nums, GPUData &d) {
    
    water_info disp;
    double norm = *(r_nums->norm);

    disp.x = *(r_nums->x) * 2 - 1.0;
    disp.y = *(r_nums->y) * 2 - 1.0;
    disp.z = *(r_nums->z) * 2 - 1.0;

    if(w->in_cell) {
        norm *= d.in_stdev;
    }
    else {
        norm *= d.out_stdev;
    }

    double nConstant = norm / sqrt(NORMSQ(disp.x, disp.y, disp.z));

    disp.x *= nConstant;
    disp.y *= nConstant;
    disp.z *= nConstant;

    return disp;
}

/**
 * Updates the provided water molecule's position based on periodic
 * boundary conditions. In effect, when a water molecule crosses any
 * edge of a periodic boundary, it simply reappears on the other side
 * of the cube. See the fmod GPU function.
 *
 * @param w     The water molecule to apply boundary conditions to
 * @param d     A GPUData class containing relevant parameters 
 */
__device__ void boundary_conditions(water_info *w, GPUData &d) {
    w->x = fmod(w->x + d.bound, d.bound);
    w->y = fmod(w->y + d.bound, d.bound);
    w->z = fmod(w->z + d.bound, d.bound);
}

/**
 * Computes and returns  the phase kick experienced at a particular (wx, wy, wz)
 * location by a water molecule. If a water molecule is inside a cell, it
 * may receive a random kick to its phase - the random kick is generated
 * using a random number supplied to this function by nD from either a uniform
 * distribution in the range [0,1) or a normal distribution with standard
 * deviation 1.0. Currently, it is implemented as a UNIFORM random double.
 *
 * @param wx        The x-coordinate to get the phase kick at 
 * @param wy        The y-coordinate to get the phase kick at 
 * @param wz        The z-coordinate to get the phase kick at
 * @param voxel     A pointer to the octree voxel that contains the (wx, wy, wz)
 *                  coordinates.
 * @param nD        Either a uniform random double in the range [0, 1) or a 
 *                  normal random double with standard deviation 1.0.
 * @param in_cell   true If the given water molecule is inside a cell, false
 *                  otherwise
 * @param d         A GPUData class containing relevant parameters
 *
 * @return A double containing the phase kick, in radians, experienced by
 *         a water molecule at the given position.
 */
inline __device__ double accumulatePhase(double &wx, double &wy, double &wz,
        gpu_node* voxel, double nD, bool in_cell, GPUData &d) { 
    double B = get_field(wx, wy, wz, voxel, d);
    double phase = 0;

#ifdef RANDOM_KICK
    // If inside a cell, add a random kick drawn from the Cauchy distribution (when the flag is defined).
    phase += (in_cell) *( d.phase_stdev*sqrt(1/(abs(nD-.5)*2)-1)*(((nD-.5)>0) - ((nD-.5))<0) * d.tau+d.phase_k*1e-3*42.58*2*M_PI*7*d.tau);
#elif defined CONSTANT_KICK
    // If inside a cell, add a constant kick (when the flag is defined)
    phase += (in_cell) * d.phase_k * d.tau;
#endif 
    phase += B * 2 * M_PI * d.g * d.tau * 1e-3;

    return phase;
}

/**
 * Global GPU kernel (can be called from a CPU function) that computes
 * and stores the positions of diffusing water molecules. All of the
 * relevant data (pointers, constants, random numbers, etc.) need
 * to be stored in the parameter GPUData structure. Each thread of this
 * kernel handles the diffusion of a single water molecule (so there
 * are 40000-10000 threads executing concurrently on the GPU).
 *
 * This kernel runs in sprints, which means it simulates the diffusion
 * of the water molecules up to a quantity <sprintSteps> timesteps. Each
 * thread has its own set of random numbers and blocks in memory allocated
 * to it. 
 * 
 * At the launch of the kernel, each thread does the following: 
 * 
 * 1. Advances its local pointers according to its thread index, so it
 *    points to the correct set of random numbers allotted to it.
 * 2. Copies its water molecule to register memory for faster computation
 * 
 * At each timestep, each thread does the following:
 *
 * 1. Gets a random displacement for its water molecule based on its current
 *    position, and then adds that displacement to the its water molecule's
 *    position.
 * 2. Applies periodic boundary conditions, making sure the water molecule
 *    stays within the simulation box
 * 3. Updates the cell nearest to itself
 * 4. Checks for reflection off a cell boundary; if a reflection takes place,
 *    the water molecule reverts to its initial position before the displacement
 *    was added.
 * 5. The new position of the water molecule is written to global memory, as
 *    well as a boolean indicating whether the water currently resides within a
 *    cell
 * 6. The local pointers for random number resources are advanced by the stride
 *    length (the total number of water molecules)
 *
 * Before the termination of the kernel, each thread copies its local register
 * water molecule back to a special array within global memory.
 *
 * At the termination of the kernel, the buffer used to store uniform random
 * numbers is now filled with the positions of water molecules over many
 * steps of diffusion
 *
 * @param d     A GPUData class that contains all of the simulation data,
 *              including pointers to random number resources, etc.
 */
__global__ void simulateDiffusion(GPUData d)  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    water_info w;

    if(tid < num_water) {
        // Copy water to local register 
        w = d.waters[tid];
        updateNearest(&w, d);

        bool* inc_ptr = d.in_cell + tid;

        // Set up random number structure
        struct t_rands r_nums;
        r_nums.x = d.x;
        r_nums.y = d.y; 
        r_nums.z = d.z; 
        r_nums.norm = d.normal_doubles;
        r_nums.coin = d.coins;
        r_nums.advancePointer(tid);

        for(int i = 0; i < d.timesteps; i++) {
            water_info init = w;

            water_info disp = rand_displacement(&w, &r_nums, d);
            w.x += disp.x;
            w.y += disp.y;
            w.z += disp.z;
            boundary_conditions(&w, d);
            updateNearest(&w, d);

            // Check cell boundary / MNP reflection

            if(cell_reflect(&init, &w, &r_nums, d)) {
                w = init;
            }

            // Store the position of the water molecule in global memory.
            *(r_nums.x) = w.x;
            *(r_nums.y) = w.y;
            *(r_nums.z) = w.z; 
            *inc_ptr = w.in_cell;

            inc_ptr += num_water;
            r_nums.advancePointer(d.num_waters); 
        }
        
    }

    // Update the running time counter
    __syncthreads();
    if(tid == 0) {
        *d.time += d.timesteps;
    }

    // Copy the register water molecule back to global memory
    if(tid < num_water) {
        d.waters[tid] = w;
    }
}

/**
 * Global GPU function that computes the phase accumulation that waters
 * experience at the locations within the x, y, and z arrays, pointers
 * to which are contained in the GPUData class. The phase kicks
 * for each water at each tstep are stored in the GPU array target.
 * The function requires an array in_cell indicating whether each
 * water molecule is inside a cell at a given timestep. It also
 * requires an array of random doubles, either distributed uniformly
 * or normally, which it uses to compute intracellular phase kicks.
 *
 * Postcondition: The target array is filled with phase kicks that each water
 * molecule experiences at each timestep over a specified range of timesteps
 *
 * @param target            The target array to store the computed phase kicks
 * @param in_cell           An array giving, at each timestep, whether a 
 *                          particular water molecule is inside a cell or not.
 * @param random_doubles    An array of random doubles used for intracellular
 *                          phase computation
 * @param length            The length of the target array, i.e. the quantity
 *                          num_water * sprintSteps, used to bound the array
 * @param d                 A reference to the GPUData object containing
 *                          the arrays of positions to compute phase kicks for,
 *                          as well as relevant parameters.
 */
__global__ void computePhaseAccumulation(
        double* __restrict__ target,
        bool* __restrict__ in_cell,
        double* __restrict__ random_doubles,
        int length, 
        GPUData d) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; 

    d.x += tid;
    d.y += tid;
    d.z += tid;
    target += tid;
    in_cell += tid;
    random_doubles += tid;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) { 
        double x = *(d.x);
        double y = *(d.y); 
        double z = *(d.z);

        // Perform the octree lookup
        gpu_node* voxel = get_voxel(x, y, z, d);
        *target = accumulatePhase(x, y, z, voxel, *random_doubles, *in_cell, d);
       
        d.x += stride;
        d.y += stride;
        d.z += stride; 
        target += stride;
        in_cell += stride;
        random_doubles += stride;
    }
}

/**
 * GPU kernel that adds a set of aggregated phase kicks to the internal phase 
 * variables of the water molecules passed in a parameter array.
 *
 * @param waters    A GPU array of water molecules
 * @param update    The array of phase kicks to add to each corresponding water
 * @param len       The number of waters in the input array
 */
__global__ void performUpdate(water_info* __restrict__ waters, 
        double* __restrict__ update, int len) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < len) {
        // Store updated phase in register for reduction
        double n_phase = waters[tid].phase + update[tid];

        // Overwrite the update with the cosine of the phase
        update[tid] = cos(n_phase);  

        // Copy new phase back to water
        waters[tid].phase = n_phase;
    }
}

/**
 * GPU kernel that flips the phases of the specified water molecules.
 */
__global__ void flipPhases(water_info* __restrict__ waters, int len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < len)
        waters[tid].phase *= -1; 
}

/**
 * Copies the nearest cell lookup table to the GPU, storing the device pointer
 * in the provided GPUData class.
 *
 * @param sourceTable   A pointer to the 2D array on the host pointing to
 *                      the nearest cell lookup table
 * @param d             A reference to the GPUData struct to store a pointer
 *                      to the GPU lookup table.
 */
void cpyLookupDevice(int **sourceTable, GPUData &d) {
    int h3 = hashDim * hashDim * hashDim;
    d.localLookup = new int*[h3];

    for(int i = 0; i < h3; i++) {
        d.localLookup[i] = (int *) cudaAllocate(maxNeighbors * sizeof(int));
        copyToDevice((void *) d.localLookup[i], (void *) sourceTable[i],
            maxNeighbors * sizeof(int));
    }
    d.lookupTable = (int**) cudaAllocate(h3 * sizeof(int**));
    copyToDevice((void *) d.lookupTable, d.localLookup,
        h3 * sizeof(int*));
}

/**
 * Simulates the diffusion and phase kicks experienced by water molecules
 * in a BacteriaBox simulation model. The function operates in the following
 * steps:
 *
 * 1. Initializes performance timers, random number generators, GPU handlers
 *    for linear algebra, etc.
 * 2. Intializes a simulation box object and populates it with waters, MNPs,
 *    and cells. It also constructs the octree in host (CPU) memory.
 * 3. Initialize a GPUData class with pointers and data to be shuttled to the
 *    GPU, and set all relevant parameters.
 * 4. Calculates how many uniform and normal random numbers are needed for
 *    each sprint, and allocates memory on the GPU for the MNPs, waters,
 *    cells, and random number resources.
 * 5. Random numbers, that are generated as one large batch by GPU functions,
 *    are partitioned out to different arrays in the GPU.
 * 6. Array of ones is created, and a thrust pointer is created for adding
 *    phase kicks together.
 * 7. MNPs, waters, cells are all uploaded to the GPU.
 * 
 * The function then executes the entire simulation of water molecules in
 * sprints of <sprintSteps> timesteps each. At each sprint:
 *
 * 1. The required number of uniform and normal random numbers are generated
 *    on the GPU.
 * 2. The diffusion path of each water molecule over the interval <sprintSteps>
 *    is calculated.
 * 3. At each of these computed locations, a second kernel is called to compute
 *    the phase kick experienced at each particular location.
 * 4. For each set of steps in the printing frequency interval:
 *      a. A matrix operation is used to add up the phase kicks for each water
 *         molecule
 *      b. A kernel computes the cosine of each of the phases, and a thrust
 *         reduction sums up and returns the net magnetization, which is then
 *         printed to a file. The current time counter is advanced.
 *
 * @param filename  The file to print the summed magnetizations at each multiple
 *                  of the printing frequency.
 */
void simulateWaters(std::string filename) {
    cout << "Starting GPU Simulation..." << endl;
    cout << "Printing to: " << filename << endl;
    ofstream fout(filename); 

    // Initialize performance timer
    Timer timer;

    // Initialize PRNG seed for MNPs and waters
    std::random_device rd;
    XORShift<uint64_t> gen(time(NULL) + rd());

    // Initialize GPU random number generator and a linear algebra handler
    gpu_rng gpu_rand_gen;
    Handler handle(0); 

    // Initialize a simulation box containing the waters, MNPs, and cells,
    // and populate it with those components
    BacteriaBox simBox(num_cells, num_water, &gen);
    simBox.populateSimulation();

    // Initialize a structure containing data to be shuttled to the GPU
    GPUData d;
    setParameters(d);
    d.num_mnps = simBox.getMNPCount(); 
    initOctree(simBox.getOctree(), d);

    // Compute number of uniform and random doubles needed for each sprint
    int totalUniform =  num_uniform_doubles * num_water * sprintSteps;
    int totalNormal = num_normal_doubles * num_water * sprintSteps;
    int initTime = 0;
 
    // GPU memory allocations performed here
    p_array<water_info> dev_waters(num_water, simBox.getWaters(), d.waters);
    p_array<Triple> dev_lattice(num_cells, simBox.getCells(), d.lattice); 

    // Partition out the uniform random numbers
    d_array<double> dev_uniform(totalUniform);
    d.x = dev_uniform.dp();
    d.y = dev_uniform.dp() + sprintSteps * num_water;
    d.z = dev_uniform.dp() + 2 * sprintSteps * num_water;
    d.coins = dev_uniform.dp() + 3 * sprintSteps * num_water;

    d_array<double> dev_normal(totalNormal, d.normal_doubles);
    p_array<int> dev_time(1, &initTime, d.time);
    d_array<double> update(num_water);
    d_array<bool> d_in_cell(num_water * sprintSteps, d.in_cell);

    // Wrap the phase update in a thrust device pointer for summation
    thrust::device_ptr<double> td_ptr = thrust::device_pointer_cast(update.dp());

    // Initialize a device array of ones for use in matrix computation 
    p_array<double> ones(pfreq);

    for(int i = 0; i < ones.getSize(); i++) {
        ones[i] = 1;
    }

    // Memory uploads to GPU performed here
    dev_lattice.deviceUpload(); 
    dev_waters.deviceUpload();
    dev_time.deviceUpload();
    ones.deviceUpload();
    cpyLookupDevice(simBox.getLookupTable(), d);
  
    int num_phase_blocks = 40000; 

    cout << "Kernel prepped!" << endl;
    cout << "Starting GPU computation..." << endl;
    timer.cpuStart();

    // Run the kernel in sprints 
    int time = 0;
    for(int i = 0; i < (t / sprintSteps); i++) {
        // Get the random numbers needed for the simulation
        gpu_rand_gen.getUniformDoubles(totalUniform, dev_uniform.dp());
        gpu_rand_gen.getNormalDoubles(totalNormal, dev_normal.dp());

        // Simulate water molecule diffusion without field computation 
        simulateDiffusion<<<num_blocks, threads_per_block>>>(d);

        // Compute the phase kick acquired by each water molecule at each location 
        computePhaseAccumulation<<<num_phase_blocks, threads_per_block>>>
            (d.coins, 
             d.in_cell, 
             dev_uniform.dp() + 4 * sprintSteps * num_water, 
             sprintSteps * num_water, 
             d);

        /** 
         * Use a matrix vector operation to add up the phase kicks
         * We will use the array of uniform doubles as a temporary buffer to 
         * store phase kicks. After each summation, we use another kernel to 
         * update the water molecules' phases and perform a memory reduction.
         */
        for(int j = 0; j < sprintSteps; j += pfreq) {
            cublasDgemv(
                handle.cublasH, 
                CUBLAS_OP_N,
                num_water, pfreq,
                &alpha,
                d.coins + j * num_water, num_water,
                ones.dp(), 1,
                &beta,
                update.dp(), 1 
                );

            // Add the accumulated phases to the water molecules' variables
            performUpdate<<<num_blocks, threads_per_block>>>(d.waters, 
                update.dp(), num_water);

            // Get the magnetization sum via thrust reduction 
            double target = thrust::reduce(td_ptr, td_ptr + num_water);

            time += pfreq;

            // If time is a multiple of Carr-Purcell time, flip the phase
            if(time % tcp == 0) {
                flipPhases<<<num_blocks, threads_per_block>>>(d.waters, num_water);
            } 

            fout << ((double) time * tau) << delim << target << endl;  
        }
    }

    // Report the elapsed time for the simulation
    float elapsedTime = timer.cpuStop();
    cout << "Kernel execution complete! Elapsed time: "
        << elapsedTime << " ms" << endl;

    // Clean up all allocated resources on the GPU, close file handle
    finalizeGPU(d);
    fout.close();
}
