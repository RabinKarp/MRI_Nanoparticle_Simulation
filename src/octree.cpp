/**
 * @author  Aadyot Bhatnagar
 * @date    August 18, 2016
 * @file    octree.cpp
 * @brief   Defines the methods relevant in constructing and traversing the
 *          octree that represents physical space. Each leaf holds the magnitude
 *          of the B_z field found at its center, or a boolean determining if it
 *          is close enough to a magnetic nanoparticle that the B field should
 *          just be calculated explicitly.
 */

#include <iostream>
#include <ctime>
#include <thread>
#include <cassert>
#include "octree.h"
#include "parameters.h"

/**
 * Given a set of (dx, dy, dz) coordinates in the space [0, bound)^3 and a
 * number (the depth) of desired precision bits per coordinate, this function
 * generates a Morton code 3*depth bits in length by converting each of dx, dy,
 * and dz to an integer in the range [0, 2^depth) and interleaving their bits
 * to generate the form {x_n, y_n, z_n, x_n-1, ... z_2, x_1, y_1, z_1}. This is
 * the order in which the octree is vectorized, so index computations can now be
 * conducted without any conditional branching.
 *
 * NOTE: nodes of the octrees have their Morton codes initialized the same way
 * for the bottom 3*depth bits, but the (3*depth + 1)'th bit is always set.
 * This allows one to distinguish between nodes at different depths. Second,
 * the most significant bit of a node's Morton code will be set IF AND ONLY
 * IF it is a leaf node. Since there is one more level of navigation after
 * reaching a leaf (determining which of its 8 B field values to use), nodes
 * can be level 20 at the depest, with their B field "children" being the
 * only points of the tree occupying the 21st level.
 */
uint64_t Octree::morton_code(int depth, double x, double y, double z)
{
#ifdef DEBUG_TREE
    assert(depth < 21);
#endif
    // generate integers in [0, 2^n) corresponding to x, y, and z and return
    // the Morton code corresponding to them
    double size = pow(2, depth);
    uint32_t idx_x = floor(x / p.bound * size);
    uint32_t idx_y = floor(y / p.bound * size);
    uint32_t idx_z = floor(z / p.bound * size);
    uint64_t answer = 0;
    // start by shifting the third byte, since we only look @ the first 21 bits
    if (depth > 16)
    {
        answer |=   morton_z[(idx_z >> 16) & 0xFF ] |
                    morton_y[(idx_y >> 16) & 0xFF ] |
                    morton_x[(idx_x >> 16) & 0xFF ];
        answer <<= 24;
    }

    // shift second byte
    if (depth > 8)
    {
        answer |=   morton_z[(idx_z >> 8) & 0xFF ] |
                    morton_y[(idx_y >> 8) & 0xFF ] |
                    morton_x[(idx_x >> 8) & 0xFF ];
        answer <<= 24;
    }

    // shift first byte
    answer |=   morton_z[(idx_z) & 0xFF ] |
                morton_y[(idx_y) & 0xFF ] |
                morton_x[(idx_x) & 0xFF ];
    return answer;
}

/**
 * The octree is constructed as an array of 8^min_depth subtrees each stored in
 * a vector to support multithreading the construction of the data structure.
 * This hybrid construction has the added benefit of adding a hashing element
 * to tree navigation, whereby finding the relevant subtree is done via an
 * arithmetic hashing function, and navigation of that subtree is done in a
 * more traditional manner. This substantially reduces the number of memory
 * accesses required for tree searches.
 */
std::vector<oct_node> *Octree::find_tree(double wx, double wy, double wz)
{
    return space + morton_code(min_depth, wx, wy, wz);
}

/**
 * Helper function to find the child index of a parent node at depth d that
 * holds the water molecule at (wx, wy, wz). This is done by taking the Morton
 * code of (wx, wy, wz) at depth d + 1 and returning the last 3 bits, which
 * would encode one additional level of depth.
 */
unsigned Octree::find_child(double wx, double wy, double wz, int d)
{
    return morton_code(d + 1, wx, wy, wz) & 7;
}

/**
 * Helper method to recursively navigate a subtree of the global octree and find
 * the node containing a given (x,y,z) coordinate. The depth of the node given
 * as an argument is d.
 */
oct_node *Octree::find_node(oct_node *n, double wx, double wy, double wz, int d)
{
#ifdef DEBUG_TREE
    uint64_t mc = morton_code(d, wx, wy, wz) | (1 << (3*(unsigned)d));
    assert(mc == ((n->mc << 1) >> 1));
#endif
    // base case -- node is a leaf
    if (n->mc >> 63)
        return n;

    // otherwise, navigate to the appropriate child and recurse
    unsigned child_no = find_child(wx, wy, wz, d);
    return find_node(n + n->child[child_no].idx, wx, wy, wz, d + 1);
}

/**
 * Returns the node of the octree a given water molecule w resides in
 */
oct_node *Octree::get_voxel(water_info *w)
{
    double wx = w->x, wy = w->y, wz = w->z;
    return find_node(find_tree(wx, wy, wz)->data(), wx, wy, wz, min_depth);
}

/**
 * Returns the pre-hashed B field stored at the node of an octree a given water
 * molecule resides in.
 */
double Octree::get_field(water_info *w, oct_node *leaf)
{
    // if not provided, get the leaf node the water resides in
    double wx = w->x, wy = w->y, wz = w->z;
    if (!leaf) leaf = get_voxel(w);

    // calculate the node's depth in the tree from its Morton code
    uint64_t depth = 0, mc = (leaf->mc << 1) >> 1;
    while (mc >>= 3) depth++;
#ifdef DEBUG_TREE
    mc = (leaf->mc << 1) >> 1;
    assert(mc == (morton_code(depth, wx, wy, wz) | (1 << 3*(unsigned)depth)));
#endif

    // use Morton code's depth to find child index to find value of B to return
    unsigned child_no = find_child(wx, wy, wz, depth);
    double B = (double)leaf->child[child_no].B;

#ifndef DEBUG_FIELD
    // add in contributions from resident MNPs zeroed out during construction
    if (leaf->resident)
    {
        std::vector<MNP_info>::iterator np;
        for (np = leaf->resident->begin(); np < leaf->resident->end(); np++) {
            B += dipole_field(wx - np->x, wy - np->y, wz - np->z, np->M);
        }
    }
#endif
    return B;
}

/**
 * Determines if a nanoparticle resides in the space encompassed by the node
 * with bottom left front corner (x, y, z) and side length g. Changed for the
 * singel cell diple code to include MNPs within a CELL radius, not a scale multiple
 * of the cluster radius.
 */
bool Octree::MNP_in_space(MNP_info *mnp, double x, double y, double z, double g)
{
    double dx = x + g/2 - mnp->x;
    double dy = y + g/2 - mnp->y;
    double dz = z + g/2 - mnp->z;
    return NORMSQ(dx, dy, dz) < pow(g/2*sqrt(3) + p.scale * mnp->r, 2);
}

/**
 * Of all the magnetic nanoparticles present in a larger source space, all
 * those contained within a smaller destinatinon space (which should be a
 * subset of the source space) are copied into that destination space. The
 * bottom left back corner & sidelength of the source destination space are
 * specified as arguments.
 */
void Octree::transfer_MNPs(std::vector<MNP_info> *orig, oct_node *dst,\
    double x, double y, double z, double g)
{
    if (!orig)
        return;
    MNP_info *mnp = orig->data();
    dst->resident = new std::vector<MNP_info>;
    for (unsigned i = 0; i < orig->size(); i++)
    {
        if (MNP_in_space(mnp, x, y, z, g))
            dst->resident->push_back(*mnp);
        mnp++;
    }
    if (dst->resident->empty())
    {
        delete dst->resident;
        dst->resident = NULL;
    }
}

/**
 * Calculates the field generated by a point dipole with strength M.
 */
double Octree::dipole_field(double dx, double dy, double dz, double M)
{
    double divisor = pow(NORMSQ(dx, dy, dz), 2.5);
    return M * 1e11 * (2*dz*dz - dx*dx - dy*dy) / divisor;
}

/**
 * Calculates the B_z field generated by all dipoles at (x, y, z).
 */
double Octree::field(double x, double y, double z)
{
    double ret = 0;
    std::vector<MNP_info>::iterator np;
    for (np = mnps->begin(); np < mnps->end(); np++)
        ret += dipole_field(x - np->x, y - np->y, z - np->z, np->M);
    return ret;
}

/**
 * Calculates the magnitude of the gradient of B_z at (x, y, z) as generated by
 * all the dipoles. Returns 0 within a certain distance of a nanoparticle.
 */
double Octree::grad(double x, double y, double z, double g)
{
    double ret_x = 0, ret_y = 0, ret_z = 0;
    std::vector<MNP_info>::iterator np;
    for (np = mnps->begin(); np < mnps->end(); np++)
    {
        double dx = x - np->x;
        double dy = y - np->y;
        double dz = z - np->z;
        double M = np->M;

        //if(NORMSQ(dx, dy, dz) < pow(sqrt(3)*g + p.scale * np->r, 2))
        //    return 0;
       
        // Modified: now if we are within a scale multiple, we don't add in the
        // gradient contribution from a given nanoparticle, but we keep the
        // contribution from the rest of the nanoparticles
        if(NORMSQ(dx, dy, dz) > pow(sqrt(3)*g + p.scale * np->r, 2)) {
            // Factor of 10^17 inserted by HD. See magnetic field calculation
            // comment for rationale
            double divisor = pow(NORMSQ(dx, dy, dz), 3.5);
            ret_x += 3e17*M*dx * (1*(dx*dx + dy*dy) - 4*dz*dz) / divisor;
            ret_y += 3e17*M*dy * (1*(dx*dx + dy*dy) - 4*dz*dz) / divisor;
            ret_z += 3e17*M*dz * (3*(dx*dx + dy*dy) - 2*dz*dz) / divisor; 
        }

    }

    return sqrt(NORMSQ(ret_x, ret_y, ret_z));
}

/**
 * Builds a subtree of the overall octree in contiguous memory. Each subtree is
 * stored in depth-first order and is constructed by generating a new vector
 * every time we need to generate a subtree at the next level. Once a leaf is
 * reached, we append the vector associated with that leaf to its parent. This
 * process goes all the way up the hierarchy until the entire subtree is stored
 * in depth-first order in the vector provided as an argument.
 */
void Octree::build_subtree(std::vector<oct_node> *curr, double max_prod,\
    double min_g, double x, double y, double z, double g)
{
    // Base case -- the cube's sidelength is smaller than the minimum allowed,
    // or the magnetic field changes sufficiently little over the cube
    if (g < min_g || (g/2 * 1e-6 * grad(x+g/2, y+g/2, z+g/2, g/2)) < max_prod)
    {
        // set most significant bit of node's Morton code to show it is a leaf
        oct_node *leaf = curr->data();
        leaf->mc |= (1ULL << 63);

        // compute B field for each sub-octant of the leaf node
        for (int i = 0; i < 8; i++)
        {
            double xn = x + g/4 + (double)((i >> 2) & 1) * g/2;
            double yn = y + g/2 + (double)((i >> 1) & 1) * g/2;
            double zn = z + g/2 + (double)((i >> 0) & 1) * g/2;
            double B = field(xn, yn, zn);
    #ifndef DEBUG_FIELD
            // Subtract out field contributions from resident MNPs -- this way,
            // we can add in more precise values when computing the B field at
            // some arbitrary point in the node
            if (leaf->resident)
            {
                std::vector<MNP_info>::iterator np;
                std::vector<MNP_info> *resident = leaf->resident;
                for (np = resident->begin(); np < resident->end(); np++)
                {
                    double dx = xn - np->x;
                    double dy = yn - np->y;
                    double dz = zn - np->z;
                    B -= dipole_field(dx, dy, dz, np->M);
                }
            }
    #endif
            leaf->child[i].B = B;
        }
    }

    // Otherwise, partition the current node of the octree further
    else
    {
        // iterate through children one-by-one for depth-first linearization
        for (int i = 0; i < 8; i++)
        {
            // create child i, the next node in the list
            oct_node *parent = curr->data();
            std::vector<oct_node> *child = new std::vector<oct_node>;
            child->emplace_back((parent->mc << 3) | i);

            // initialize nanoparticles for child i
            double xn = x + (double)((i >> 2) & 1) * g/2;
            double yn = y + (double)((i >> 1) & 1) * g/2;
            double zn = z + (double)((i >> 0) & 1) * g/2;
            transfer_MNPs(parent->resident, child->data(), xn, yn, zn, g/2);

            // recursively build up subtree for child i in depth-first order
            build_subtree(child, max_prod, min_g, xn, yn, zn, g/2);
            parent->child[i].idx = curr->size();
            curr->insert(curr->end(), child->begin(), child->end());
            delete child;
        }

        // internal nodes don't need to keep track of their resident MNPs
        oct_node *parent = curr->data();
        if (parent->resident)
        {
            delete parent->resident;
            parent->resident = NULL;
        }
    }
}

/**
 * Each thread of execution will construct one subtree at a time for the global
 * octree representing all of the physical space we are interested in. Each
 * subtree will have its root at level min_depth of the global octree. All
 * 8^min_depth subtrees will be constructed as vectors stored in depth-first
 * order. These vectors will themselves be stored in an array. Thus, tree
 * traversal will have two steps: determining which subtree to start in, and
 * then traversing that subtree to reach the desired node.
 */
void Octree::init_subtrees(int tid, double max_prod, double min_g)
{
    int arr_size = (int)pow(8, min_depth);
    for (int offset = tid; offset < arr_size; offset += p.num_threads)
    {
        double x = 0, y = 0, z = 0, g = p.bound;
        int temp = offset;

        // find (x, y, z) & g for the root of the subtree @ space[offset]
        for (int j = min_depth - 1; j >= 0; j--)
        {
            int oct_id = temp / (int)pow(8, j); // between 0 and 7
            double dx = (double)(oct_id >> 2) * g/2;
            double dy = (double)((oct_id >> 1) & 1) * g/2;
            double dz = (double)(oct_id & 1) * g/2;
            x += dx; y += dy; z += dz;
            temp %= (int)pow(8, j); // between 0 and 8^j
            g /= 2;
        }

        // build up the subtree at space[offset]
        std::vector<oct_node> *root = space + offset;
        root->emplace_back(morton_code(min_depth, x, y, z));
        root->data()->mc |= (1 << (3*min_depth));
        transfer_MNPs(this->mnps, root->data(), x, y, z, g);
        build_subtree(root, max_prod, min_g, x, y, z, g);
    }
}

/**
 * The constructor builds the octree seeded with magnetic nanoparticles. The
 * value n dictates how many threads will be used to build the tree, i.e. 8^n
 * threads will be used. The tree itself is stored as an array of vectors, where
 * each vector stores a subtree whose root has a granularity just under max_g.
 *
 * Though complex, there are good reasons for such a data structure. First, the
 * array of trees structure allows us to add a hashing behavior to the tree to
 * reduce the overall number of memory accesses entailed in tree iteration.
 * Second, the hashing function substantially reduces execution divergence when
 * being used on a GPU to search the tree (compared to a traditional tree
 * search). Third, keeping an array of multiple vectors representing subtrees
 * reduces the time required in building the data structure, since we are no
 * longer concerned with a large number of coalescing steps.
 */
Octree::Octree(double max_prod, double max_g, double min_g, XORShift<> &gen,\
        std::vector<MNP_info> *mnps)
{
    // Initialize magnetic nanoparticles & modify parameters if necessary
    this->mnps = mnps;
#ifdef DEBUG_FIELD
    max_prod = -1; max_g = bound; min_g = bound/50;
#elif defined EXPLICIT
    max_prod = -1; max_g = bound/1.5; min_g = max_g;
#endif

    // Initialize the array of vectors (hashtable part of the tree)
    this->min_depth = ceil(log(p.bound / max_g) / log(2));
    this->max_depth = ceil(log(p.bound / min_g) / log(2));
    int arr_size = (int)pow(8, min_depth);
    this->space = new std::vector<oct_node>[arr_size];
    std::cout << "After applying boundary conditions, there are ";
    std::cout << mnps->size() << " nanoparticles." << std::endl;
    std::cout << "The tree has a minimum depth of " << min_depth << " and a ";
    std::cout << "maximum depth of " << max_depth << "." << std::endl;

    // Initialize the vectorized subtrees corresponding to each hashtable entry
    std::vector<std::thread> threads;
    for (int i = 0; i < p.num_threads; i++)
        threads.emplace_back(&Octree::init_subtrees, this, i, max_prod, min_g);
    for (auto &t : threads)
        t.join();

    // Add up the number of nodes in the tree as a whole
    uint64_t num_nodes = 0;
    uint64_t occ = 0;
    for (int i = 0; i < arr_size; i++)
    {
        num_nodes += this->space[i].size();
        std::vector<oct_node>::iterator j;
        for (j = space[i].begin(); j < space[i].end(); j++)
            if (j->resident) occ++;
    }

    double max = 0;
    for (int i = min_depth; i <= max_depth; i++) max += pow(8, i);
    std::cout << "The tree has " << (double)num_nodes << " nodes, or ";
    std::cout << (double)num_nodes/max*100  << "% of " << (double)max << " ";
    std::cout << "possible." << std::endl << "In total, " << occ << " nodes, ";
    std::cout << "or " << (double)occ/num_nodes*100 << "% of all the nodes in ";
    std::cout << "the tree hold MNPs." << std::endl << std::endl;

#ifdef DEBUG_FIELD
    // if we are debugging the field, the tree is constructed uniformly, and an
    // output file containing the value of B_z at every node is generated
    std::ofstream out_file;
    out_file.open("T2_sim_tree_field_info.csv");
    for (int i = 0; i < arr_size; i++)
    {
        std::vector<oct_node>::iterator j;
        for (j = space[i].begin(); j < space[i].end(); j++)
            if (j->mc >> 63)
                for (int k = 0; k < 8; k++)
                    out_file << j->child[k].B << std::endl;
    }
    out_file.close();
#endif
}

/**
 * Deletes the vector of magnetic nanoparticles and each of the vectors
 * representing subtrees of the larger global octree.
 */
Octree::~Octree()
{
    for (int i = 0; i < (int)pow(8, min_depth); i++)
    {
        std::vector<oct_node>::iterator j;
        for (j = space[i].begin(); j < space[i].end(); j++)
            if (j->resident) delete j->resident;
    }
    delete[] space;
}
