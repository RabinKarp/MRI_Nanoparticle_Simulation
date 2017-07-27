#ifndef GPU_OCTREE
#define GPU_OCTREE

#include "cuda_helpers.h"
#include "octree.h"
#include "rand_walk.h"

typedef struct gpu_node {
    uint64_t mc;        // Mocton code of node; leaf if leftmost bit is set
    B_idx child[8];     // child offsets (internal) or child B fields (leaves)
    int numResidents;
    int resIdx;
} gpu_node;

void initOctree(Octree *oct, struct GPUData &d);
__device__ double get_field(water_info *w, struct GPUData &d);

#endif
