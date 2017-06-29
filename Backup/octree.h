/*
 * @author  Aadyot Bhatnagar
 * @date    June 24, 2016
 * @file    octree.h
 * @brief   Header file defining the structures used to construct and maitnain
 *          an octree stored in contiguous memory as a vector.
 */

#ifndef OCTREE_H
#define OCTREE_H

#include "rand_walk.h"
#include <cstdlib>
#include <vector>
#include <cmath>

const int num_mnps = 500;
const int num_water = 200;
const double mnp_radius = .1;

using namespace std;

/* A node in the octree used to partition space representing a cube with the
 * specified center and granularity. To be stored in a vector.
 */
typedef struct oct_node
{
    double x, y, z; // coordinates of center
    double g;       // granularity of node
    double B=0;     // strength of longitudinal B field

    // offsets from children in the vector
    unsigned c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0;

    // initialize with given (x,y,z) as center and granularity g
    oct_node(double x, double y, double z, double g) : x(x), y(y), z(z), g(g) {}
} oct_node;

/* An intermediary structure used when building the octree */
typedef struct temp_node
{
    vector<oct_node> *space;
    MNP_node *resident=NULL;
} temp_node;

/* The class encapsulating the entire octree. The relevant traversals are
 * handled from this class via public method calls.
 */
class Octree
{
    public:
    Octree(double max_product, double max_g, double min_g, int thread_id,\
            int n=0);
    ~Octree();

    /* Vector of magnetic nanoparticles */
    vector<MNP_info> *mnps;
    vector<oct_node> **space;

    /* Magnetic field profiles */
    double field(double x, double y, double z);
    double grad(double x, double y, double z, double g=0);
    double part_field(double x, double y, double z, double g=0);

    /* Return the node a given water molecule resides in */
    oct_node *get_voxel(water_info *w);

    private:
    double bound;
    int nesting;

    /* Helpers for the constructor */
    temp_node *init_root(double dim, double x=0, double y=0, double z=0);
    vector<oct_node> *build(temp_node *curr, double max_product, double max_g,\
            double min_g);
    void build_subtree(int thread_no, double max_product, double max_g,\
            double min_g);

    /* Helpers for build() */
    void transfer_mnp_nodes(temp_node *src, temp_node *dst);
    bool MNP_in_space(MNP_info *mnp, oct_node *n);
    void octants(vector<oct_node> *next_space, vector<oct_node> *curr_space,\
            int i);

    /* Operations on linked lists of MNP nodes */
    void append_MNP(MNP_info *mnp, temp_node *space);
    void delete_MNP_nodes(MNP_node *mnp);

    /* Private magnetic field profiles */
    double part_grad(double x, double y, double z, double g=0);
    typedef double (Octree::*octree_fn)(double x, double y, double z, double g);
    double B_function(double x, double y, double z, octree_fn f, double g=0);

    /* Helpers for octree traversal */
    vector<oct_node> *find_tree(double wx, double wy, double wz);
    oct_node *find_node(oct_node *space, double wx, double wy, double wz);
};

#endif /* OCTREE_H */
