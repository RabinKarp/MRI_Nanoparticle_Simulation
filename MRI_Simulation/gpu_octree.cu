#include <iostream>
#include <cuda.h>
#include "parameters.h"
#include "gpu_octree.h"
#include "cuda_helpers.h"

typedef struct gpu_node {
    uint64_t mc;        // Mocton code of node; leaf if leftmost bit is set
    B_idx child[8];     // child offsets (internal) or child B fields (leaves)
    int numResidents;
    int resIdx;
} oct_node;

/**
 * Initialize a GPU verison of the octree from the CPU version.
 */
void initOctree(Octree *oct, GPUData &d) {
    // Initialize octree parameters

    vector<oct_node> *vec_nodes = oct->space;

    gpu_node *localTree = new gpu_node[vec_nodes->size()];
    MNP_info *localMNPs = new MNP_info[oct->mnps->size()];

    int mnp_idx = 0;
    for(int i = 0; i < vec_nodes->size(); i++) {
        oct_node current = (*vec_nodes)[i];
        localTree[i].mc = current.mc;

        for(int j = 0; j < 8; j++) {
            localTree[i].child[j] = current.child[j];
        }

        localTree[i].numResidents = current.mnps->size();
        localTree[i].resIdx = mnp_idx;

        for(int j = 0; j < current.mnps->size(); j++) {
            localMNPs[mnp_idx] = (*(current.mnps))[j];
            mnp_idx++;
        }
    }

    HANDLE_ERROR(cudaMalloc((void **) &(d.tree),
        vec_nodes->size() * sizeof(oct_node)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.mnps),
        oct->mnps->size() * sizeof(MNP_info)));

    // Allocate arrays for morton codes
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_x),
        256 * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_y),
        256 * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &(d.morton_z),
        256 * sizeof(uint32_t)));

    HANDLE_ERROR(cudaMemcpy(d.tree, localTree,
        vec_nodes->size() * sizeof(oct_node),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.mnps, localMNPs,
        oct->mnps->size() * sizeof(MNP_info),
        cudaMemcpyHostToDevice));

    // Copy over the Morton code arrays
    HANDLE_ERROR(cudaMemcpy(d.morton_x, morton_x,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.morton_y, morton_y,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d.morton_z, morton_z,
        256 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    delete[] localTree;
    delete[] localMNPs;
}

__device__ double dipole_field(double dx, double dy, double dz, double M)
{
    double d2x = dx*dx;
    double d2y = dy*dy;
    double d2z = dz*dz;
    double sum = d2x + d2y + d2z;
    double divisor = sum * sum * sqrt(sum);
    return M * 1e11 * (2*d2z - d2x - d2y) / divisor;
}

__device__ morton_code(int depth, double x, double y, double z, GPUData &d) {

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

__device__ oct_node* find_tree(double wx, double wy, double wz, GPUData &d)
{
    return d.tree + morton_code(d.min_depth, wx, wy, wz);
}

/*
 * Helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 */
__device__ unsigned find_child(double wx, double wy, double wz, int d)
{
    return morton_code(d + 1, wx, wy, wz) & 7;
}

// TODO: Check this function!
__device__ oct_node* find_node(oct_node *n, double wx, double wy, double wz, int d) {
    while( ! (n->mc >> 63)) {
        // otherwise, navigate to the appropriate child and recurse
        unsigned child_no = find_child(wx, wy, wz, d);
        d += 1;
        n += n->child[child_no].idx;
    }
    return n;
}

__device__ oct_node* get_voxel(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    return find_node(find_tree(wx, wy, wz, d), wx, wy, wz, d.min_depth);
}

/**
 * Returns the B field at the location of a particular water molecule
 */
__device__ double get_field(water_info *w, GPUData &d) {
    double wx = w->x, wy = w->y, wz = w->z;
    oct_node *leaf = get_voxel(w, d);
    __syncthreads();

    uint64_t depth = 0, mc = (leaf->mc << 1) >> 1;
    while (mc >>= 3) depth++;

    // use Morton code's depth to find child index to find value of B to return
    unsigned child_no = find_child(wx, wy, wz, depth);
    double B = (double)leaf->child[child_no].B;

    // add in contributions from resident MNPs zeroed out during construction
    for(int i = leaf->resIdx; i < leaf->resIdx + leaf->num_mnps; i++) {
        MNP_info *np = d.mnps + i;
        B += dipole_field(wx - np->x, wy - np->y, wz - np->z, np->M);
    }
    __syncthreads();

    return B;
}
