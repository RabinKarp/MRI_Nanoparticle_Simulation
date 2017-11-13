/**
 * @author  Aadyot Bhatnagar
 * @author  Vivek Bharadwaj
 * @date    September 23, 2017
 * @file    rand_walk.h
 * @brief   Contains information about a water molecule and some of the basic
 *          operations that may be performed on it.
 */

#ifndef RAND_WALK_H
#define RAND_WALK_H
#include <vector>
#include <cmath>
#include "parameters.h"
#include "xorshift.h"
#define NORMSQ(x, y, z) (x)*(x)+(y)*(y)+(z)*(z)     // compute the norm-squared

/**
 * This structure contains information about a water molecule, including its
 * position in (x,y,z) coordinates, the number of times it has crossed the
 * periodic x boundary (and in which direction), and the phase of its
 * transverse magnetization. Information about its residence in a cell in
 * its lattice is also included.
 */
typedef struct water_info {
    double x, y, z;
    double phase=0;
    bool in_cell;
    bool in_mnp;
    int nearest;
    double field;
  public:
    /**
     * The addition of a water_info struct w to this entails adding the (x,y,z)
     * coordinates of w to the coordinates of this.
     */
    void operator+=(struct water_info &w)
    {
        this->x += w.x;
        this->y += w.y;
        this->z += w.z;
    }

    /**
     * The -= operation behaves in essentially the same way as the += operator.
     */
    void operator-=(struct water_info &w)
    {
        this->x -= w.x;
        this->y -= w.y;
        this->z -= w.z;
    }

} water_info;

/** 
 * Describes a magnetic nanoparticle 
 */
typedef struct MNP_info
{
    double x, y, z; // coordinates of center
    double r;       // radius
    double M;       // magnetic moment

    // initialize with given (x,y,z) as center, radius r, and magnetic moment M
    MNP_info(double x, double y, double z, double r, double M):
        x(x), y(y), z(z), r(r), M(M) {}

    MNP_info() {};
} MNP_info;

#endif /* RAND_WALK_H */
