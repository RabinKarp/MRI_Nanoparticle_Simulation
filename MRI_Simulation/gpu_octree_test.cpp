#include "parameters.h"
#include "fcc_diffusion.h"
#include "cuda_helpers.h"
#include "fcc_diffusion.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <map>

using namespace std;

void setParameters(GPUData &d) {
    // Initialize constants for the GPU
    d.in_stdev = sqrt(pi * D_cell * tau);
    d.out_stdev = sqrt(pi * D_extra * tau);

    d.reflectIO = 1 - sqrt(tau / (6*D_cell)) * 4 * P_expr;
    d.reflectOI = 1 - ((1 - d.reflectIO) * sqrt(D_cell/D_extra));

    d.cell_r = cell_r;
    d.bound = bound;
}

double dipole_field(double dx, double dy, double dz, double M)
{
    double divisor = pow(NORMSQ(dx, dy, dz), 2.5);
    return M * 1e11 * (2*dz*dz - dx*dx - dy*dy) / divisor;
}

uint64_t morton_code(int depth, double x, double y, double z, GPUData &d) {

    double size = pow(2, depth);
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

gpu_node** find_tree(double wx, double wy, double wz, GPUData &d)
{
    return d.tree + morton_code(d.min_depth, wx, wy, wz, d);
}

/*
 * Helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 */
unsigned find_child(double wx, double wy, double wz, int d, GPUData &data)
{
    return morton_code(d + 1, wx, wy, wz, data) & 7;
}

// TODO: Check this function!
gpu_node* find_node(gpu_node *n, double wx, double wy, double wz, int d, GPUData &data) {
    // base case -- node is a leaf
    if (n->mc >> 63)
        return n;

    // otherwise, navigate to the appropriate child and recurse
    unsigned child_no = find_child(wx, wy, wz, d, data);
    return find_node(n + n->child[child_no].idx, wx, wy, wz, d + 1, data);
}

gpu_node* get_voxel(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    return find_node(*(find_tree(wx, wy, wz, d)), wx, wy, wz, d.min_depth, d);
    //return find_tree(wx, wy, wz, d);
}

/**
 * Returns the B field at the location of a particular water molecule
 */
double get_field(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    gpu_node *leaf = get_voxel(w, d);

    uint64_t depth = 0, mc = (leaf->mc << 1) >> 1;
    while (mc >>= 3) depth++;

    // use Morton code's depth to find child index to find value of B to return
    unsigned child_no = find_child(wx, wy, wz, depth, d);
    double B = (double)leaf->child[child_no].B;

    // add in contributions from resident MNPs zeroed out during construction
    for(int i = 0; i < leaf->numResidents; i++) {
        MNP_info *np = leaf->resident + i;
        B += dipole_field(wx - np->x, wy - np->y, wz - np->z, np->M);
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
                  localTree[i][j].resident = (MNP_info*) malloc(sizeof(MNP_info) * localTree[i][j].numResidents);
                  // Copy MNPs to device
                  memcpy(localTree[i][j].resident, current[j].resident->data(), sizeof(MNP_info) * localTree[i][j].numResidents);
              }
              else {
                  localTree[i][j].numResidents = 0;
                  localTree[i][j].resident = nullptr;
              }
          }
          // Store the subtree size locally
          d.sizes[i] = current.size();
          // Now copy the entire subtree to the device, storing in the local pointers struct
          localPointers[i] = (gpu_node *) malloc(sizeof(gpu_node) * current.size());
          memcpy(localPointers[i], localTree[i], sizeof(gpu_node) * current.size());
        }

        // Now copy the entire tree into memory
        d.tree = (gpu_node**) malloc(sizeof(gpu_node**) * arr_size);
        memcpy(d.tree, localPointers, sizeof(gpu_node**) * arr_size);
    }

    d.morton_x = (uint32_t*) malloc(256 * sizeof(uint32_t));
    d.morton_y = (uint32_t*) malloc(256 * sizeof(uint32_t));
    d.morton_z = (uint32_t*) malloc(256 * sizeof(uint32_t));


    // Wrong memory copies
    memcpy(d.morton_x, morton_x, 256 * sizeof(uint32_t));
    memcpy(d.morton_y, morton_y, 256 * sizeof(uint32_t));
    memcpy(d.morton_z, morton_z, 256 * sizeof(uint32_t));

    /*for(int i = 0; i < vec_nodes.size(); i++) {
        cout << (*vec_nodes)[i].mc << " " << (d.tree + i)->mc << endl;

    }*/

    for(int i = 0; i < arr_size; i++) {
        delete[] localTree[i];
    }
}

void destroyTree(GPUData &d) {
    for(int i = 0; i < d.arr_size; i++) {
        if(d.tree[i]) {
            for(int j = 0; j < d.sizes[i]; j++) {
                // First destroy the MNPs held onto by each node
                if(d.tree[i][j].resident)
                    free(d.tree[i][j].resident);
            }
            // Next free the subtree
            free(d.tree[i]);
        }
    }

    // Free the morton code arrays
    free(d.morton_x);
    free(d.morton_y);
    free(d.morton_z);

    // Host pointer, can call delete
    delete [] d.sizes;
}

int main() {
    std::random_device rd;
    XORShift<uint64_t> gen(time(NULL) + rd());

    FCC lattice(D_cell, D_extra, P_expr);
    vector<MNP_info> *mnps = lattice.init_mnps(gen);

    double max_product = 2e-6, max_g = 5, min_g = .002;
    uint64_t sTime = time(NULL);
    Octree tree(max_product, max_g, min_g, gen, mnps);
    uint64_t eTime = time(NULL) - sTime;
    std::cout << "Octree took " << eTime / 60 << ":";
    if (eTime % 60 < 10) std::cout << "0";
    std::cout << eTime % 60 << " to build." << std::endl << std::endl;

    GPUData d;
    setParameters(d);
    initOctree(&tree, d);
    water_info test;

    srand (time(NULL) + rd());

    for(int i = 0; i < 1000; i++) {
      test.x = (0.0 + rand()) / RAND_MAX * bound;
      test.y = (0.0 + rand()) / RAND_MAX * bound;
      test.z = (0.0 + rand()) / RAND_MAX * bound;

      cout << test.x << " " << test.y << " " << test.z << endl;
      cout << tree.get_field(&test) << endl;
      cout << get_field(&test, d) << endl;
    }
    destroyTree(d);
}
