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

#ifndef BACTERIA_BOX_H 
#define BACTERIA_BOX_H 

#include "SimulationBox.h"

/**
 * A BacteriaBox is an implementation of SimulationBox that throws
 * cells of a specified radius randomly within the simulation boundary;
 * places a magnetic dipole of 0 radius and specified magnetic moment at the
 * center of each cell; and sets the water molecules to start in a sub-cube
 * centered within the larger simulation cube. 
 */
class BacteriaBox: public SimulationBox 
{
public:
    BacteriaBox(XORShift<> *gen);
    
    virtual ~BacteriaBox();
     
private:
    // All implementation details are left to virtual function overrides 
    virtual void init_cells()   override;
    virtual void init_mnps()    override;
    virtual void init_waters()  override;

};

#endif 
