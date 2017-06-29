/*
 * @author  Aadyot Bhatnagar
 * @date    June 21, 2016
 * @file    rand_walk.h
 * @brief   Contains information about a water molecule and some of the basic
 *          operations that may be performed on it.
 */

#ifndef RAND_WALK_H
#define RAND_WALK_H
#include <vector>
#include <random>
#include "xorshift.h"
#define NORMSQ(x, y, z) x*x+y*y+z*z

using namespace std;

/*
 * This structure contains information about a water molecule, including its
 * position in (x,y,z) coordinates, the number of times it has crossed the
 * periodic x boundary (and in which direction), and the phase of its
 * transverse magnetization. Information about its residence in a cell in
 * its lattice is also included.
 */
typedef struct water_info {
    double x=0, y=0, z=0;
    double phase = 0;
    int cross_x = 0;
    int nearest;
    bool in_cell;

    /*
     * The addition of a water_info struct w to this entails adding the (x,y,z) 
     * coordinates of w to the coordinates of this.
     */
    void operator+=(struct water_info &w)
    {
        this->x += w.x;
        this->y += w.y;
        this->z += w.z;
    }

    /*
     * The -= operation behaves in essentially the same way as the += operator.
     */
    void operator-=(struct water_info &w)
    {
        this->x -= w.x;
        this->y -= w.y;
        this->z -= w.z;
    }

} water_info;

/* Describes a magnetic nanoparticle */
typedef struct MNP_info
{
    double x, y, z; // coordinates of center
    double r;       // radius
    double M;       // magnetic moment

    // initialize with given (x,y,z) as center, radius r, and magnetic moment M
    MNP_info(double x, double y, double z, double r, double M):
        x(x), y(y), z(z), r(r), M(M) {}
} MNP_info;

/* Linked list node for magnetic nanoparticles */
typedef struct MNP_node
{
    MNP_info *mnp;
    struct MNP_node *next=NULL;
} MNP_node;

water_info rand_displacement(normal_distribution<> normal, XORShift<> &gen);
water_info *init_molecules(double dim, int num, vector<MNP_info> *mnps, XORShift<> &gen);

#endif /* RAND_WALK_H */
