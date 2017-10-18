/**
 * @author  Vivek Bharadwaj 
 * @date    September 17, 2017
 * @file    BacteriaBox.h 
 * @brief   Header file for the BacteriaBox class, a subclass of SimulationBox 
 *          which initializes the simulation by throwing random cells of a 
 *          specified radius within the simulation bound; placing a magnetic 
 *          dipole of 0 radius and specified magnetic moment at the center 
 *          of each cell; and placing a water molecules in a subcube of
 *          side length smaller than the total simulation bound in the
 *          center of the larger simulation bound cube.
 */

#ifndef FCC_BOX_H 
#define FCC_BOX_H 

#include "SimulationBox.h"

/**
 * A BacteriaBox is an implementation of SimulationBox that throws
 * cells of a specified radius randomly within the simulation boundary;
 * places a magnetic dipole of 0 radius and specified magnetic moment at the
 * center of each cell; and sets the water molecules to start in a sub-cube
 * centered within the larger simulation cube. 
 */
class FCCBox: public SimulationBox 
{
public:
    FCCBox(XORShift<> *gen);
    
    virtual ~FCCBox();
     
private:
    // All implementation details are left to virtual function overrides 
    virtual void init_cells()   override;
    virtual void init_mnps()    override;
    virtual void init_waters()  override;

	water_info rand_displacement(double norm, XORShift<> *gen);

    /*
     * Instance variable representing the centers of all the cells in an FCC
     * lattice (unscaled).
     */
    double fcc[172][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},\
        {0,0,-1},{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},\
        {-1,-1,-1},{-1,-1,1},{-2,-2,-1},{-2,-2,1},{-2,-1,-2},{-2,-1,0},\
        {-2,-1,2},{-2,0,-1},{-2,0,1},{-2,1,-2},{-2,1,0},{-2,1,2},{-2,2,-1},\
        {-2,2,1},{-1,-2,-2},{-1,-2,0},{-1,-2,2},{-1,0,-2},{-1,0,2},{-1,2,-2},\
        {-1,2,0},{-1,2,2},{0,-2,-1},{0,-2,1},{0,-1,-2},{0,-1,2},{0,1,-2},\
        {0,1,2},{0,2,-1},{0,2,1},{1,-2,-2},{1,-2,0},{1,-2,2},{1,0,-2},{1,0,2},\
        {1,2,-2},{1,2,0},{1,2,2},{2,-2,-1},{2,-2,1},{2,-1,-2},{2,-1,0},{2,-1,2},\
        {2,0,-1},{2,0,1},{2,1,-2},{2,1,0},{2,1,2},{2,2,-1},{2,2,1},{-2,-2,3},\
        {-2,-2,-3},{-2,0,3},{-2,0,-3},{-2,2,3},{-2,2,-3},{-2,3,-2},{-2,-3,-2},\
        {-2,3,0},{-2,-3,0},{-2,3,2},{-2,-3,2},{-1,-1,3},{-1,-1,-3},{-1,1,3},\
        {-1,1,-3},{-1,3,-1},{-1,-3,-1},{-1,3,1},{-1,-3,1},{0,-2,3},{0,-2,-3},\
        {0,0,3},{0,0,-3},{0,2,3},{0,2,-3},{0,3,-2},{0,-3,-2},{0,3,0},{0,-3,0},\
        {0,3,2},{0,-3,2},{1,-1,3},{1,-1,-3},{1,1,3},{1,1,-3},{1,3,-1},{1,-3,-1},\
        {1,3,1},{1,-3,1},{2,-2,3},{2,-2,-3},{2,0,3},{2,0,-3},{2,2,3},{2,2,-3},\
        {2,3,-2},{2,-3,-2},{2,3,0},{2,-3,0},{2,3,2},{2,-3,2},{3,-2,-2},\
        {-3,-2,-2},{3,-2,0},{-3,-2,0},{3,-2,2},{-3,-2,2},{3,-1,-1},{-3,-1,-1},\
        {3,-1,1},{-3,-1,1},{3,0,-2},{-3,0,-2},{3,0,0},{-3,0,0},{3,0,2},{-3,0,2},\
        {3,1,-1},{-3,1,-1},{3,1,1},{-3,1,1},{3,2,-2},{-3,2,-2},{3,2,0},{-3,2,0},\
        {3,2,2},{-3,2,2},{1,-3,-3},{1,-3,3},{1,3,-3},{1,3,3},{-1,-3,-3},\
        {-1,-3,3},{-1,3,-3},{-1,3,3},{3,-1,-3},{3,-1,3},{-3,-1,-3},{-3,-1,3},\
        {3,1,-3},{3,1,3},{-3,1,-3},{-3,1,3},{3,-3,-1},{3,3,-1},{-3,-3,-1},\
        {-3,3,-1},{3,-3,1},{3,3,1},{-3,-3,1},{-3,3,1},{-3,-3,-3},{-3,-3,3},\
        {-3,3,-3},{-3,3,3},{3,-3,-3},{3,-3,3},{3,3,-3},{3,3,3}};
};

#endif 
