# MRI_Nanoparticle_Simulation

GPU accelerated Monte Carlo simulation for diffusing water molecules experiencing phase kicks due to field inhomogeneities. 

This version of the code is implemented such that intracellular water molecules receive some random phase kick according to a normal distribution with a specified standard deviation.

## How to run the simulation:
Two easy steps: make and submit the subfile - details below.

### To make on the Penguin on Demand server:
Call `make` in the `src` directory. You do NOT need to load in any dependant modules manually - this is taken care of when you call `make` by the modules.sh shell script. Consequently, if you wish to load in some new custom modules, modify this file.

It is recommended to run `make clean` first to ensure that all header files are properly updated. In addition, you can run `make clean_output` to delete all CSV files in the directory `src`. Calling `make clean` does NOT perform this behavior.

### To run the simulation:
A subfile is already configured, `GPU_Test.sub`. Simply call and run this simulation with the appropriate walltime defined. This simulation takes about 14 minutes for 4000 water molecules simulating 40 ms of diffusion - expect the runtime to double if either the simulation time or the number of water molecules is doubled.

### Where to find the outputs: 
Check the src folder - eventually, I will add a line in the code to send outputs to their own dedicated folder.

### How to modify the file-naming convention:
Modify the file `T2_GPU_sim.cu`. This file contains nothing else except a function to set the output filename and the main method that initializes the water molecule simulation.

### NOTE:
This simulation can only be run on the queue H30G (that has K40 GPUs installed). In addition, at least 8 cores per node must be configured in order to use a single GPU - each node supports up to 16. 

## Relevant parameters in this new simulation:

**mmoment:** The magnetic moment for each cell’s dipole

**phase_stdev:** The standard deviation of the random phase kick that intracellular water molecules receive.

**totaltime:** The total time to run the simulation for, in ms

**num_cells:** The number of cells to throw randomly in the simulation bound.

**cell_r:** The radius of each cell to throw, in microns

**bound:** The size of the entire simulation box, in microns

**water_start_bound:** All of the water molecules begin the simulation in a box with this dimension in the center of the larger simulation space.

**totalTime:** The total time in ms to run the simulation for, but it’s an integer count of ms now (easier with the GPU architecture).

**num_water:** Sets the number of waters in the simulation, currently set at 4000, but with a caveat: increasing the number of waters without decreasing sprintSteps might cause the GPU to run out of memory. The simulation is safe for 8000 water molecules and 25,000 sprint steps, and probably for 10,000 water molecules. We can handle more than 10,000 waters simply by decreasing sprintSteps, a parameter which simply controls the number of timeseteps that the GPU processes in one batch (it continues to run batches until all time steps are simulated.k

### WARNINGS:

As of this update, it is possible to reduce the timestep, but this requires changing the code in more than one place -i.e. not just the parameter file. I can handle this easily upon request and am working to modify the code so you can modify the timestep directly from the parameter files.

