#include "utilities/utilities.h"
#include "cuda_sim.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string>

#include <vector>
#include <fstream>
#include <thread>
#include <iostream>
#include "math.h"

#include "parameters.h"
#include "fcc_diffusion.h"
#include "gpu_random.h"
#include "octree.h"


using namespace std;

struct t_rands {
    double *x;
    double *y;
    double *z;
    double *coin;
    double *norm;

    __device__ void advancePointer(int stride) {
        x += stride;
        y += stride;
        z += stride;
        coin += stride;
        norm += stride;
    }
};

const int num_blocks = (num_water + threads_per_block - 1) / threads_per_block;

const double g = 42.5781e6;             // gyromagnetic ratio in MHz/T
const double pInt = 1e-3;
const int pfreq = (int)(pInt/tau);      // print net magnetization every 1us

// Constant for lin. alg. operations
const double alpha = 1;
const double beta = 0;

#define num_uniform_doubles 5 // # of uniform doubles per tstep used to generate random direction 
#define num_normal_doubles 1  // # of normal  doubles per water per tstep

//==============================================================================
// Octree-related functions
//==============================================================================

__device__ double dipole_field(double dx, double dy, double dz, double M, GPUData &d)
{
    double sqDist = NORMSQ(dx, dy, dz);
    double divisor = sqDist * sqDist * sqrt(sqDist);
    return (sqDist > d.cell_r * d.cell_r) * M * 1e11 * (2*dz*dz - dx*dx - dy*dy) / divisor;
}

__device__ uint64_t morton_code(int depth, double &x, double &y, double &z, GPUData &d) {
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

__device__ gpu_node** find_tree(double wx, double wy, double wz, GPUData &d)
{
    return d.tree + morton_code(d.min_depth, wx, wy, wz, d);
}

/*
 * Helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 */
__device__ unsigned find_child(double wx, double wy, double wz, int d, GPUData &data)
{
    return morton_code(d + 1, wx, wy, wz, data) & 7;
}

// TODO: Check this function!
__device__ gpu_node* find_node(gpu_node *n, double wx, double wy, double wz, int d, GPUData &data) {
    // base case -- node is a leaf
    if (n->mc >> 63)
        return n;

    // otherwise, navigate to the appropriate child and recurse
    unsigned child_no = find_child(wx, wy, wz, d, data);
    return find_node(n + n->child[child_no].idx, wx, wy, wz, d + 1, data);
}

__device__ gpu_node* get_voxel(double &wx, double &wy, double &wz, GPUData &d) {
    return find_node(*(find_tree(wx, wy, wz, d)), wx, wy, wz, d.min_depth, d);
}

/**
 * Returns the B field at the location of a particular water molecule
 */
__device__ double get_field(double &wx, double &wy, double &wz, gpu_node* leaf, GPUData &d) {
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
 * Initialize a GPU verison of the octree from the CPU version.
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
}

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

void finalizeGPU(GPUData &d) {
    destroyTree(d);
}

void setParameters(GPUData &d) {
    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);

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
// Simulation functions
//==============================================================================

__device__ void updateNearest(water_info *w, GPUData &d) {
    double cubeLength = d.bound / hashDim;
    int x_idx = w->x / cubeLength;
    int y_idx = w->y / cubeLength;
    int z_idx = w->z / cubeLength;

    int* nearest =
        d.lookupTable[z_idx * d.hashDim * d.hashDim
        + y_idx * d.hashDim
        + x_idx];

    double cDist = d.bound * d.bound * 3;
    int cIndex = -1;
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

inline __device__ bool cell_reflect(water_info *i, water_info *f, t_rands *r_nums, GPUData &d) { 
    double coin = *(r_nums->coin); 
    bool flip = (i->in_cell && (! f->in_cell) && coin < d.reflectIO)
                    || ((! i->in_cell) && f->in_cell && coin < d.reflectOI);
    return flip;
}

__device__ bool mnp_reflect(water_info *w, MNP_info *mnp, int num_mnps, GPUData &d) {
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

inline __device__ water_info rand_displacement(water_info *w, t_rands *r_nums, GPUData &d) {
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

__device__ void boundary_conditions(water_info *w, GPUData &d) {
    w->x = fmod(w->x + d.bound, d.bound);
    w->y = fmod(w->y + d.bound, d.bound);
    w->z = fmod(w->z + d.bound, d.bound);
}

__device__ double accumulatePhase(double &wx, double &wy, double &wz,
        gpu_node* voxel, double nD, bool in_cell, GPUData &d) { 
    double B = get_field(wx, wy, wz, voxel, d);
    double phase = 0;

#ifdef RANDOM_KICK
    // If inside a cell, add a random kick (when the flag is defined)
    phase += (in_cell) * nD * d.phase_stdev*d.tau+d.phase_k*d.tau;
#elif defined CONSTANT_KICK
    // If inside a cell, add a constant kick (when the flag is defined)
    phase += (in_cell) * d.phase_k * d.tau;
#endif 
    phase += B * 2 * M_PI * d.g * d.tau * 1e-3;

    return phase;
}
// END PHASE ACCUMULATION FUNCTIONS


__global__ void simulateWaters(GPUData d)  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    water_info w;

    if(tid < num_water) {
        // Copy water to local register 
        w = d.waters[tid];
        updateNearest(&w, d);

        bool* inc_ptr = d.in_cell + tid;

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

__global__ void computePhaseAccumulation(
        double* __restrict__ target,
        bool* __restrict__ in_cell,
        double* __restrict__ normal_doubles,
        int length, 
        GPUData d) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; 

    d.x += tid;
    d.y += tid;
    d.z += tid;
    target += tid;
    in_cell += tid;
    normal_doubles += tid;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) { 
        double x = *(d.x);
        double y = *(d.y); 
        double z = *(d.z); 
    
        gpu_node* voxel = get_voxel(x, y, z, d);
        *target = accumulatePhase(x, y, z, voxel, *normal_doubles, *in_cell, d);
       
        d.x += stride;
        d.y += stride;
        d.z += stride; 
        target += stride;
        in_cell += stride;
        normal_doubles += stride;
    }
}

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

__global__ void flipPhases(water_info* __restrict__ waters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    waters[tid].phase *= -1; 
}

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

void destroyLookupDevice(GPUData &d) {
    for(int i = 0; i < hashDim * hashDim * hashDim; i++) {
        cudaFree(d.localLookup[i]);
    }
    cudaFree(d.lookupTable);
    delete[] d.localLookup;
}

void simulateWaters(std::string filename) {
    cout << "Starting GPU Simulation..." << endl;
    cout << "Printing to: " << filename << endl;
    ofstream fout(filename); 

    // Initialize PRNG seed for MNPs and waters
    std::random_device rd;
    XORShift<uint64_t> gen(time(NULL) + rd());

    // Initialize GPU random number generator and a linear algebra handler
    gpu_rng gpu_rand_gen;
    Handler handle(0); // TODO: Fix this in the multi-GPU case!

    // The simulation has 3 distinct components: the lattice, the water
    // molecules, and the nanoparticles

    FCC lattice(D_cell, D_extra, P_expr, gen);
    vector<MNP_info> *mnps = lattice.init_mnps();
    water_info *waters = lattice.init_molecules(num_water,gen);
    Triple* linLattice = lattice.linearLattice();

    // Initialize the octree
    double max_product = 2e-6, max_g = 5, min_g = .002;
    uint64_t sTime = time(NULL);
    Octree tree(max_product, max_g, min_g, gen, mnps);
    uint64_t eTime = time(NULL) - sTime;
    std::cout << "Octree took " << eTime / 60 << ":";
    if (eTime % 60 < 10) std::cout << "0";
    std::cout << eTime % 60 << " to build." << std::endl << std::endl;

    // Sort the water molecules by their morton codes
    // this can speed up performance by a factor of 1-2 !
    lattice.sortWaters(waters, num_water, tree);

    GPUData d;
    setParameters(d);
    d.num_mnps = mnps->size();
    initOctree(&tree, d);
    cout << "Allocated GPU Octree!" << endl;

    // Compute number of uniform and random doubles needed for each sprint
    int totalUniform =  num_uniform_doubles * num_water * sprintSteps;
    int totalNormal = num_normal_doubles * num_water * sprintSteps;
    int initTime = 0;
 
    // GPU memory allocations performed here
    p_array<water_info> dev_waters(num_water, waters, d.waters);

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

    // Wrap the update in a thrust device pointer
    thrust::device_ptr<double> td_ptr = thrust::device_pointer_cast(update.dp());

    // Initialize an array containing only 1's, upload it to the GPU for use in matrix computation
    p_array<double> ones(pfreq);

    for(int i = 0; i < ones.getSize(); i++) {
        ones[i] = 1;
    } 
    ones.deviceUpload();

    p_array<Triple> dev_lattice(num_cells, linLattice, d.lattice); 
    cpyLookupDevice(lattice.lookupTable, d);
    p_array<double> dev_magnetizations(num_blocks * (t / pfreq), nullptr, d.magnetizations);

    // Initializes performance timer
    Timer timer;

    // Initialization memory copies to GPU performed here
    dev_lattice.deviceUpload(); 
    dev_waters.deviceUpload();
    dev_time.deviceUpload();

    int num_phase_blocks = 40000; 

    cout << "Kernel prepped!" << endl;
    cout << "Starting GPU computation..." << endl;
    timer.cpuStart();

    // Run the kernel in sprints due to memory limits and timeout issues
    int time = 0;
    for(int i = 0; i < (t / sprintSteps); i++) {
        gpu_rand_gen.getUniformDoubles(totalUniform, dev_uniform.dp());
        gpu_rand_gen.getNormalDoubles(totalNormal, dev_normal.dp());

        // Generate a list of positions for water molecules without computing the field
        simulateWaters<<<num_blocks, threads_per_block>>>(d);

        // Compute the phase kick acquired by each water molecule at each location 
        computePhaseAccumulation<<<num_phase_blocks, threads_per_block>>>
            (d.coins, 
             d.in_cell, 
             dev_uniform.dp() + 4 * sprintSteps * num_water, 
             sprintSteps * num_water, 
             d);

        // Use a matrix vector operation to add up the phase kicks
        // We will use the array of uniform doubles as a temporary buffer to store phase kicks
        // After each summation, we use another kernel to update the water molecules' phases and
        // perform a memory reduction
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

            performUpdate<<<num_blocks, threads_per_block>>>(d.waters, update.dp(), num_water);

            // Get the magnetization sum via thrust reduction 
            double target = thrust::reduce(td_ptr, td_ptr + num_water);

            time += pfreq;

            if(time % tcp == 0) {
                flipPhases<<<num_blocks, threads_per_block>>>(d.waters);
            } 

            fout << ((double) time * tau) << delim << target << endl;  
        }
    }

    float elapsedTime = timer.cpuStop();

    cout << "Kernel execution complete! Elapsed time: "
        << elapsedTime << " ms" << endl;


    destroyLookupDevice(d);
    finalizeGPU(d);

    delete[] linLattice;
    delete[] waters;
    delete mnps;
    fout.close();
}
