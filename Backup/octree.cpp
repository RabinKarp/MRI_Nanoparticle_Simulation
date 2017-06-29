/*
 * @author  Aadyot Bhatnagar
 * @date    June 24, 2016
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
#include "octree.h"
#include "fcc_diffusion.h"

#define SCALE 2

/*
 * Initializes a temp_node with the root of the octree and a linked list that
 * contains all relevant MNPs in the sample. The root of the octree is a cube
 * of side length 2 * dim and center (x,y,z).
 */
temp_node *Octree::init_root(double dim, double x, double y, double z)
{
    vector<oct_node> *space = new vector<oct_node>;
    space->emplace_back(x, y, z, dim);
    temp_node *root = new temp_node;
    root->space = space;
    MNP_info *np = mnps->data();
    for (unsigned i = 0; i < mnps->size(); i++)
    {
        if (MNP_in_space(np, space->data()))
            append_MNP(np, root);
        np++;
    }
    return root;
}

/*
 * Determines if a nanoparticle resides in the space encompassed by a given
 * octree node.
 */
bool Octree::MNP_in_space(MNP_info *mnp, oct_node *n)
{
    double dx = mnp->x - n->x;
    double dy = mnp->y - n->y;
    double dz = mnp->z - n->z;
    return NORMSQ(dx, dy, dz) < pow(n->g*sqrt(3) + SCALE*mnp->r, 2);
}

/*
 * Deletes the linked list of MNP_nodes associated with a temp_node.
 */
void Octree::delete_MNP_nodes(MNP_node *mnp)
{
    if (!mnp)
        return;
    delete_MNP_nodes(mnp->next);
    delete mnp;
    return;
}

/*
 * Appends an MNP_node to the linked list of MNP_nodes in a temp_node.
 */
void Octree::append_MNP(MNP_info *mnp, temp_node *space)
{
    MNP_node *next = new MNP_node;
    next->mnp = mnp;
    next->next = space->resident;
    space->resident = next;
}

/*
 * Of all the magnetic nanoparticles present in a larger source space, all
 * those contained within a smaller destinatinon space (which should be a
 * subset of the source space) are copied into that destination space.
 */
void Octree::transfer_mnp_nodes(temp_node *src, temp_node *dst)
{
    oct_node *dst_space = dst->space->data();
    MNP_node *orig = src->resident;

    while (orig)
    {
        MNP_info *mnp = orig->mnp;
        if (MNP_in_space(mnp, dst_space))
            append_MNP(mnp, dst);
        orig = orig->next;
    }
}

/*
 * Given a source space (curr_space) and a destination space (next_space), this
 * method generates the ith octant of the source space in the destination space.
 */
void Octree::octants(vector<oct_node> *curr_space, vector<oct_node> *next_space,\
        int i)
{
    int curr_size = curr_space->size();
    oct_node *first = curr_space->data();
    double x = first->x, y = first->y, z = first->z, g = first->g;
    
    switch(i)
    {
        case 1: // (+, +, +) octant
            next_space->emplace_back(x + g/2, y + g/2, z + g/2, g/2);
            curr_space->data()->c1 = curr_size;
            break;
        case 2: // (+, +, -) octant
            next_space->emplace_back(x + g/2, y + g/2, z - g/2, g/2);
            curr_space->data()->c2 = curr_size;
            break;
        case 3: // (+, -, -) octant
            next_space->emplace_back(x + g/2, y - g/2, z - g/2, g/2);
            curr_space->data()->c3 = curr_size; 
            break;
        case 4: // (+, -, +) octant
            next_space->emplace_back(x + g/2, y - g/2, z + g/2, g/2);
            curr_space->data()->c4 = curr_size;
            break;
        case 5: // (-, -, +) octant
            next_space->emplace_back(x - g/2, y - g/2, z + g/2, g/2);
            curr_space->data()->c5 = curr_size;
            break;
        case 6: // (-, +, +) octant
            next_space->emplace_back(x - g/2, y + g/2, z + g/2, g/2);
            curr_space->data()->c6 = curr_size;
            break;
        case 7: // (-, +, -) octant
            next_space->emplace_back(x - g/2, y + g/2, z - g/2, g/2);
            curr_space->data()->c7 = curr_size;
            break;
        case 8: // (-, -, -) octant
            next_space->emplace_back(x - g/2, y - g/2, z - g/2, g/2);
            curr_space->data()->c8 = curr_size;
            break;
    }
}

/*
 * Builds the octree in contiguous memory (stored in a vector). The octree is
 * stored in depth-first order and is constructed by generating a new vector
 * every time we need to generate a subtree. Once a leaf is reached, we
 * append the vector associated with that leaf to its parent. This process
 * goes all the way up the linked list/tree structure until the entire tree
 * is stored in depth-first order in the vector.
 * 
 * The secondary structure encapsulating the vectors (the temp_node) is used
 * to allow the leaves to determine whether or not they are close enough to a
 * magnetic nanoparticle for a water molecule residing in that node to have to
 * evaluate B explicitly at every time step, or whether we should pre-hash an
 * approximate value of B to that node for future evaluations.
 */
vector<oct_node> *Octree::build(temp_node *curr, double max_product,\
        double max_g, double min_g)
{
    vector<oct_node> *curr_space = curr->space;
    oct_node *first = curr_space->data();
    double x = first->x, y = first->y, z = first->z, g = first->g;

    // base case -- node is a leaf
    if ((g < min_g) || (g < max_g && g * 1e-6 * part_grad(x, y, z, g) < max_product))
    // HD Edited: Check that this is consistent with the rest of your code. I
    // instered a factor of 10^-6 to get the product in teslas (we need T/m 
    // multiplied by meters
    {
        if (!curr->resident)
            first->B = field(x, y, z);
    }
    
    // if we have not reached a small enough granularity and are still not below
    // the minimum granularity, we need to partition the current node of the
    // octree further
    else
    {
        for (int i = 1; i <= 8; i++)
        {
            // create the next node in the list
            temp_node *next = new temp_node;
            vector<oct_node> *next_space = new vector<oct_node>;
            next->space = next_space;

            // generate each of the 8 octants/subtrees
            octants(curr_space, next_space, i);

            // recursively build up each of the 8 octants into its own subtree
            // and then append it onto the current octree vector
            transfer_mnp_nodes(curr, next);
            next_space = build(next, max_product, max_g, min_g);
            curr_space->insert(curr_space->end(), next_space->begin(),\
                    next_space->end());
            delete next_space;
        }
    }
    delete_MNP_nodes(curr->resident);
    delete curr;
    return curr_space;
}


/*
 * Returns the subtree (from the array of vectors) in which the specified point
 * can be found. Sub-octants are stored in an order like (1,1),(1,2),...(8,7),
 * (8,8), i.e. each large octant occupies a contiguous eighth of array, and
 * each smaller sub-octant occupies a contiguous eighth in its encompassing
 * octant.
 */
vector<oct_node> *Octree::find_tree(double wx, double wy, double wz)
{
    double x = 0;
    double y = 0;
    double z = 0;
    double g = this->bound;
    int index = 0;

    // determine the index of the array of vectors we want to navigate to
    for (int i = nesting - 1; i >= 0; i--)
    {
        // determine which sub-octant to navigate to
        if (wx > x) {
            if (wy > y) {
                if (wz > z) { // (+, +, +), octant 1
                    x += g/2; y += g/2; z += g/2;
                    index += 0 * (int)pow(8, i);
                }
                else {        // (+, +, -), octant 2
                    x += g/2; y += g/2; z -= g/2;
                    index += 1 * (int)pow(8, i);
                }
            } 
            else {
                if (wz < z) { // (+, -, -), octant 3
                    x += g/2; y -= g/2; z -= g/2;
                    index += 2 * (int)pow(8, i);
                }
                else {        // (+, -, +), octant 4
                    x += g/2; y -= g/2; z += g/2;
                    index += 3 * (int)pow(8, i);
                }
            }
        }
        else {
            if (wz > z) {
                if (wy < y) { // (-, -, +), octant 5
                    x -= g/2; y -= g/2; z += g/2;
                    index += 4 * (int)pow(8, i);
                }
                else {        // (-, +, +), octant 6
                    x -= g/2; y += g/2; z += g/2;
                    index += 5 * (int)pow(8, i);
                }
            }
            else {
                if (wy > y) { // (-, +, -), octant 7
                    x -= g/2; y += g/2; z -= g/2;
                    index += 6 * (int)pow(8, i);
                }
                else {        // (-, -, -), octant 8
                    x -= g/2; y -= g/2; z -= g/2;
                    index += 7 * (int)pow(8, i);
                }
            }
        }
        g /= 2;
    }
    return space[index];
}

/*
 * Helper method to recursively navigate a subtree of the global octree and find
 * the node containing a given (x,y,z) coordinate.
 */
oct_node *Octree::find_node(oct_node *space, double wx, double wy, double wz)
{
    // base case -- node is a leaf
    if (space->c1 == 0)
        return space;

    // otherwise, navigate to the appropriate child
    else
    {
        double x = space->x;
        double y = space->y;
        double z = space->z;

        // determine which octant to navigate to
        if (wx > x)
        {
            if (wy > y)
            {
                if (wz > z) // (+, +, +), octant 1
                    return find_node(space + space->c1, wx, wy, wz);
                else        // (+, +, -), octant 2
                    return find_node(space + space->c2, wx, wy, wz);
            } 
            else
            {
                if (wz < z) // (+, -, -), octant 3
                    return find_node(space + space->c3, wx, wy, wz);
                else        // (+, -, +), octant 4
                    return find_node(space + space->c4, wx, wy, wz);
            }
        }

        else
        {
            if (wz > z)
            {
                if (wy < y) // (-, -, +), octant 5
                    return find_node(space + space->c5, wx, wy, wz);
                else        // (-, +, +), octant 6
                    return find_node(space + space->c6, wx, wy, wz);
            }
            else
            {
                if (wy > y) // (-, +, -), octant 7
                    return find_node(space + space->c7, wx, wy, wz);
                else        // (-, -, -), octant 8
                    return find_node(space + space->c8, wx, wy, wz);
            }
        }
    }
}

/*
 * Returns the voxel of the octree that a given water molecule resides in.
 */
oct_node *Octree::get_voxel(water_info *w)
{
    double wx = w->x;
    double wy = w->y;
    double wz = w->z;
    return find_node(find_tree(wx, wy, wz)->data(), wx, wy, wz);
}

/*
 * Each thread of execution will construct its own subtree of the global octree
 * representing all of the physical space we are interested in. Each subtree
 * will have its root at level n of the global octree, where n is the level of
 * nesting specified in the constructor for the octree. All 8^n subtrees will
 * be constructed as vectors stored in depth-first order. These vectors will
 * themselves be stored in an array. Thus, tree traversal will have two steps:
 * determining which subtree to start in, and then traversing that subtree
 * to reach the desired node.
 */
void Octree::build_subtree(int thread_no, double max_product, double max_g,\
        double min_g)
{
    double g = 3 * sqrt(2) * cell_r;
    int temp = thread_no;
    double x = 0, y = 0, z = 0;
    for (int i = nesting - 1; i >= 0; i--)
    {
        int oct_id = temp / (int)pow(8, i);
        switch(oct_id)
        {
            case 0: // (+, +, +) octant 1
               x += g/2; y += g/2; z += g/2;
               break;
           case 1: // (+, +, -) octant 2
               x += g/2; y += g/2; z -= g/2;
               break;
            case 2: // (+, -, -) octant 3
               x += g/2; y -= g/2; z -= g/2;
               break;
            case 3: // (+, -, +) octant 4
               x += g/2; y -= g/2; z += g/2;
               break;
            case 4: // (-, -, +) octant 5
               x -= g/2; y -= g/2; z += g/2;
               break;
            case 5: // (-, +, +) octant 6
               x -= g/2; y += g/2; z += g/2;
               break;
            case 6: // (-, +, -) octant 7
               x -= g/2; y += g/2; z -= g/2;
               break;
            case 7: // (-, -, -) octant 8
               x -= g/2; y -= g/2; z -= g/2;
               break;
        }
        temp %= (int)pow(8, i);
        g /= 2;
    }
    temp_node *root = init_root(g, x, y, z);
    space[thread_no] = build(root, max_product, max_g, min_g);
}

/*
 * The constructor builds the octree seeded with magnetic nanoparticles. The
 * nesting level dictates how many threads will be used to build the tree, i.e.
 * 8^n threads will be used (one for each of the relevant subtrees). These sub-
 * trees are stored as an array of vectors in (essentially) depth-first order.
 */
Octree::Octree(double max_product, double max_g, double min_g, int thread_id, int n)
{
    this->bound = cell_r * 3 * sqrt(2);

    // intialize PRNG
    XORShift<> gen(time(NULL));
    for (int i = 0; i < thread_id; i++)
        gen.jump();

    // initialize magnetic nanoparticles
    FCC lattice(cell_r, R_io, R_oi, 0);
    this->mnps = lattice.init_mnps(gen);
    //this->mnps = init_mnps(1e-15, mnp_radius, 5, gen);
    cout << mnps->size() << " nanoparticles in the lattice." << endl;

    // initialize octree representation of space
    this->nesting = n;
    int arr_size = (int)pow(8, n);
    this->space = new vector<oct_node>*[arr_size];

    // use a different thread to generate each subtree
    vector<thread> threads;
    for (int i = 0; i < arr_size; i++)
        threads.emplace_back(&Octree::build_subtree, this, i, max_product,\
                max_g, min_g);
    for (int i = 0; i < arr_size; i++)
        threads[i].join();
}

/*
 * Deletes the vector of magnetic nanoparticles and each of the vectors
 * representing subtrees of the larger global octree.
 */
Octree::~Octree()
{
    delete mnps;
    for (int i = 0; i < (int)pow(8,nesting); i++)
        delete space[i];
    delete[] space;
}

/*
 * Calculate magnitude of B_z and  magnitude of gradient of B_z while accounting
 * for periodic boundary conditions.
 */
double Octree::grad(double x, double y, double z, double g)
{
    return B_function(x, y, z, &Octree::part_grad, g);
}
double Octree::field(double x, double y, double z)
{
    return B_function(x, y, z, &Octree::part_field);
}

/*
 * TODO: ADD UP INDIVIDUAL VECTOR CONTRIBUTIONS TO GRADIENT AND TAKE THE
 * MAGNITUDE OF THE SUM! DON'T JUST SUM UP THE MAGNITUDES OF ALL THE VECTOR
 * CONTRIBUTIONS THEMSELVES!
 */

#include <mutex>
mutex mtx;

/*
 * Calculates the magnitude of the gradient of B_z at (x, y, z) generated by
 * a single lattice. Returns 0 within a certain distance of a magnetic
 * nanoparticle.
 */
double Octree::part_grad(double x, double y, double z, double g)
{
    double ret_x = 0, ret_y = 0, ret_z = 0;
    for (vector<MNP_info>::iterator i = mnps->begin(); i < mnps->end(); i++)
    {
        double dx = x - i->x;
        double dy = y - i->y;
        double dz = z - i->z;
        double M = i->M;
        double divisor = pow(NORMSQ(dx, dy, dz), 3.5);

        if (NORMSQ(dx, dy, dz) < pow(sqrt(3)*g + SCALE*i->r, 2))
            return 0;

        // Factor of 10^17 inserted by HD. See magnetic field calculation 
        // comment for rationale
        ret_x += 3e17*M*dx * (dx*dx + dy*dy - 4*dz*dz) / divisor;
        ret_y += 3e17*M*dy * (dx*dx + dy*dy - 4*dz*dz) / divisor;
        ret_z += 3e17*M*dz * (3*(dx*dx + dy*dy) - 2*dz*dz) / divisor;
    }

    return sqrt(NORMSQ(ret_x, ret_y, ret_z));
}


/*
 * Calculates the B_z field generated by one lattice at (x, y, z).
 */
double Octree::part_field(double x, double y, double z, double g)
{
    double ret = 0;
    for (vector<MNP_info>::iterator i = mnps->begin(); i < mnps->end(); i++)
    {
        double dx = x - i->x;
        double dy = y - i->y;
        double dz = z - i->z;
        // EDIT: Inserted by HD to fix field calculation. Rescaled to meters
        // from microns with 10^-12/10^-30=10^18 and inserted permeability of
        // free space/pi=(10^-7) to calculate field in T
        if (NORMSQ(dx, dy, dz) > i->r * i->r)
            ret += i->M * 1e11 * (2*dz*dz - dx*dx - dy*dy) / pow(NORMSQ(dx, dy, dz), 2.5);
    }
    return ret;
}

/*
 * Calculates the magnitude of some function of the magnetic field (either the
 * field itself, or its gradient, though this could be expanded) at (x, y, z)
 * accounting for periodic boundary conditions.
 */
double Octree::B_function(double x, double y, double z, octree_fn f, double g)
{
    double total_field = 0;
    total_field += (this->*f)(x, y, z, g);

    if (total_field == 0)
        return 0;

    if (x + 5 > bound) // close to front side
    {
        total_field += (this->*f)(x - 2*bound, y, z, 0);

        if (y + 5 > bound) // close to front and right sides
        {
            total_field += (this->*f)(x - 2*bound, y - 2*bound, z, 0);
            total_field += (this->*f)(x, y - 2*bound, z, 0);

            if (z + 5 > bound) // close to front, right, and top sides
            {
                total_field += (this->*f)(x - 2*bound, y - 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x - 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y - 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to front, right, and bottom sides
            {
                total_field += (this->*f)(x - 2*bound, y - 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x - 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y - 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }
        
        else if (y - 5 < -bound) // close to front and left sides
        {
            total_field += (this->*f)(x - 2 * bound, y + 2 * bound, z, 0);
            total_field += (this->*f)(x, y + 2*bound, z, 0);

            if (z + 5 > bound) // close to front, left, and top sides
            {
                total_field += (this->*f)(x - 2*bound, y + 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x - 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y + 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to front, left, and bottom sides
            {
                total_field += (this->*f)(x - 2*bound, y + 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x - 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y + 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }

        else // not close to left or right sides, but close to front side
        {
            if (z + 5 > bound) // close to front and top sides
            {
                total_field += (this->*f)(x - 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }
            else if (z - 5 < -bound) // close to front and bottom sides
            {
                total_field += (this->*f)(x - 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }
    }

    else if (x - 5 < -bound) // close to back side
    {
        total_field += (this->*f)(x + 2*bound, y, z, 0);

        if (y + 5 > bound) // close to back and right sides
        {
            total_field += (this->*f)(x + 2*bound, y - 2*bound, z, 0);
            total_field += (this->*f)(x, y - 2*bound, z, 0);

            if (z + 5 > bound) // close to back, right, and top sides
            {
                total_field += (this->*f)(x + 2*bound, y - 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x + 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y - 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to back, right, and bottom sides
            {
                total_field += (this->*f)(x + 2*bound, y - 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x + 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y - 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }
        
        else if (y - 5 < -bound) // close to back and left sides
        {
            total_field += (this->*f)(x + 2 * bound, y + 2 * bound, z, 0);
            total_field += (this->*f)(x, y + 2*bound, z, 0);

            if (z + 5 > bound) // close to back, left, and top sides
            {
                total_field += (this->*f)(x + 2*bound, y + 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x + 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y + 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to back, left, and bottom sides
            {
                total_field += (this->*f)(x + 2*bound, y + 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x + 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y + 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }

        else // not close to left or right sides, but close to back side
        {
            if (z + 5 > bound) // close to back and top sides
            {
                total_field += (this->*f)(x + 2*bound, y, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }
            else if (z - 5 < -bound) // close to back and bottom sides
            {
                total_field += (this->*f)(x + 2*bound, y, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }
    }

    else // not close to back or front sides
    {
        if (y + 5 > bound) // close to right side
        {
            total_field += (this->*f)(x, y - 2*bound, z, 0);

            if (z + 5 > bound) // close to top and right sides
            {
                total_field += (this->*f)(x, y - 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to bottom and right sides
            {
                total_field += (this->*f)(x, y - 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }
        
        else if (y - 5 < -bound) // close to left side
        {
            total_field += (this->*f)(x, y + 2*bound, z, 0);

            if (z + 5 > bound) // close to top and left sides
            {
                total_field += (this->*f)(x, y + 2*bound, z - 2*bound, 0);
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            }

            else if (z - 5 < -bound) // close to bottom and left sides
            {
                total_field += (this->*f)(x, y + 2*bound, z + 2*bound, 0);
                total_field += (this->*f)(x, y, z + 2*bound, 0);
            }
        }

        else // not close to left, right, front, or back sides
        {
            if (z + 5 > bound) // close to top side
                total_field += (this->*f)(x, y, z - 2*bound, 0);
            else if (z - 5 < -bound) // close to bottom side
                total_field += (this->*f)(x, y, z + 2*bound, 0);
        }
    }

    return total_field;
}
