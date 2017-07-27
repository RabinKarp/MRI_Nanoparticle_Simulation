/*
 * @author  Aadyot Bhatnagar
 * @author  Vivek Bharadwaj
 * @date    July 11, 2017
 * @file    fcc_diffusion.cpp
 * @brief   Implementation details for a class that represents a 3x3x3 face-
 *          centered cubic lattice of cells with periodic boundary conditions.
 *          The only public method is one that simulates one step of the
 *          diffusion of a water molecule through the cell lattice.
 */

#include <cmath>
#include <iostream>
#include <fstream>
#include "parameters.h"
#include "fcc_diffusion.h"
#include "rand_walk.h"

#define CELL_TYPES 6

/**
 * Computes the centers of all the cells in the FCC lattice and stores
 * them in the FCC class's member array. Also computes the array of neighbors
 * for each cell.
 */
void FCC::initLattice(int dim) {

#ifdef DEBUG_LATTICE
    std::ofstream f1("lattice.csv");
    std::ofstream f2("neighbors.csv");
#endif

    int x, y, z;
    x = 0;
    y = 0;
    z = 0;

    // Initialize every entry in the sphere lookup array to sentinel -1
    for(int i = 0; i < lw; i++) {
        for(int j = 0; j < lw; j++) {
            for(int k = 0; k < lw; k++) {
                sphereLookup[i][j][k] = -1;
            }
        }
    }

    for(int i = 0; i < num_cells; i++) {
        fcc[i][0] = x;
        fcc[i][1] = y;
        fcc[i][2] = z;
        sphereLookup[x][y][z] = i;
        x += 2;
        if(x >= lw) {
            x %= lw;
            y += 1;
            if(y >= lw) {
                y = 0;
                z += 1;
            }
        }
    }

    // Post-processing (for backwards-compatibility) subtract off the
    // lattice dimension from each of the coordinates to center the lattice
    // about the origin
    for(int i = 0; i < num_cells; i++) {
        fcc[i][0] -= dim;
        fcc[i][1] -= dim;
        fcc[i][2] -= dim;
#ifdef DEBUG_LATTICE
        f1 << fcc[i][0] << "," << fcc[i][1] << "," << fcc[i][2] << std::endl;
#endif
    }
}

/**
 * Returns an array containing triple structs with the lattice point centers
 */
Triple* FCC::linearLattice() {
    Triple *linLattice = new Triple[num_cells];

    for(int i = 0; i < num_cells; i++) {
        linLattice[i].x = fcc[i][0];
        linLattice[i].y = fcc[i][1];
        linLattice[i].z = fcc[i][2];
    }
    return linLattice;
}

int* FCC::linearLookupTable() {
    int* linLookup = new int[lw * lw * lw];
    for(int i = 0; i < lw; i++) {
        for(int j = 0; j < lw; j++) {
            for(int k = 0; k < lw; k++) {
                linLookup[i * lw * lw + j * lw + k] = sphereLookup[i][j][k];
            }
        }
    }
    return linLookup;
}

/*
 * Uses cell radius to scale coordinates of cell centers in FCC lattice and
 * initializes the scattering probabilities upon collision with a cell membrane,
 * both from the inside and from the outside. Also intializes the standard
 * deviation expected from the Gaussian distribution used to generate random
 * displacement vectors.
 */
FCC::FCC(double D_in, double D_out, double P_expr)
{
    initLattice(n);
    for (int i = 0; i < num_cells; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fcc[i][j] *= cell_r * sqrt(2) * fcc_pack;
            fcc[i][j] += bound/2;
        }
    }

    norm_in = std::normal_distribution<>(0, sqrt(pi * D_in * tau));
    norm_out = std::normal_distribution<>(0, sqrt(pi * D_out * tau));
    this->reflectIO = 1 - sqrt(tau / (6*D_in)) * 4 * P_expr;
    this->reflectOI = 1 - ((1 - reflectIO) * sqrt(D_in/D_out));
}


/*
 * Given a set of Cartesian boundary conditions, initialize a given number of
 * water molecules distributed randomly over the cubical space [0, L)^3.
 */
water_info *FCC::init_molecules(double L, int n, std::vector<MNP_info> *mnps,\
    XORShift<> &gen)
{
    water_info *molecules = new water_info[n];
    water_info *temp = molecules;

    for (int i = 0; i < n; i++)
    {
        double x = gen.rand_pos_double() * L;
        double y = gen.rand_pos_double() * L;
        double z = gen.rand_pos_double() * L;
        bool inside = false;

        // Make sure the molecule isn't inside a nanoparticle.
        std::vector<MNP_info>::iterator np;
        for (np = mnps->begin(); np < mnps->end() && !inside; np++)
        {
            double dx = x - np->x;
            double dy = y - np->y;
            double dz = z - np->z;
            if (NORMSQ(dx, dy, dz) < np->r * np->r)
                inside = true;
        }

        // If the molecule IS inside a nanoparticle, generate it again.
        if (inside)
        {
            i--;
            continue;
        }

        // Otherwise, store it in the array and move on.
        temp->x = x;
        temp->y = y;
        temp->z = z;
        temp->phase = 0;
        this->update_nearest_cell_full(temp);
        temp++;
    }

    std::cerr << "Molecules initialized!" << std::endl;
    return molecules;
}

/**
 * Helper function that checs if a sphere with specified x, y, and z
 * coordinates and specified radius overlaps with any of the nanoparticles
 * in a given vector.
 */
bool FCC::checkMNPOverlap(std::vector<MNP_info> *mnps,
    double x, double y, double z, double r) {
    bool overlaps = false;
    std::vector<MNP_info>::iterator curr;
    for (curr = mnps->begin(); curr != mnps->end() && !overlaps; curr++)
    {
        double dx = x - curr->x;
        double dy = y - curr->y;
        double dz = z - curr->z;
        if (NORMSQ(dx, dy, dz) < pow(r + curr->r, 2))
          overlaps = true;
    }
    return overlaps;
}

/**
 * Helper function that checks if the sphere with the given (x, y, z)
 * coordinates and radius OVERLAPS with the boundary of any sphere in
 * the FCC lattice.
 */
bool FCC::checkLatticeOverlap(double x, double y, double z, double r) {
    bool overlaps = false;
    for(int i = 0; i < num_cells; i++) {
        double dx = x - fcc[i][0];
        double dy = y - fcc[i][1];
        double dz = z - fcc[i][2];

        double sqDist = NORMSQ(dx, dy, dz);

        if (sqDist < pow(cell_r + r, 2)
            && sqDist > pow(cell_r - r, 2)) {
            overlaps = true;
        }
    }
    return overlaps;
}

/**
 * Helper function that checks if the given (x, y, z) coordinate is
 * contained in any cell of the FCC lattice. Returns the index of the lattice
 * cell that contains the given point, or -1 if no cell contains that point.
 */
inline int FCC::checkLatticeContainment(double x, double y, double z) {
    int containCell = -1;
    for(int i = 0; i < num_cells; i++) {
      double dx = x - fcc[i][0];
      double dy = y - fcc[i][1];
      double dz = z - fcc[i][2];

      if(NORMSQ(dx, dy, dz) < pow(cell_r, 2)) {
        containCell = i;
      }
    }
    return containCell;
}

/**
 * Handle the case of unclustered MNP's in extracellular or intracellular
 * space, or both
 */
#ifdef UNCLUSTERED
/*
 * Initializes the specified number of individual, unclustered magnetic
 * nanoparticles
 */
std::vector<MNP_info> *FCC::init_mnps(XORShift<> &gen)
{
    std::vector<MNP_info> *mnps = new std::vector<MNP_info>;
    for (int i = 0; i < num_mnps; i++)
    {
        double x, y, z;
        double r = mnp_radius;
        bool contained = false;
        bool overlaps = false;
        bool invalid = true;

        // keep generating (x,y,z) coordinates until we get one
        // that satisfies the parameters and doesn't overlap with other objects
        // in the simulation
        while (invalid)
        {
            x = gen.rand_pos_double() * bound;
            y = gen.rand_pos_double() * bound;
            z = gen.rand_pos_double() * bound;
            invalid = false;
            contained = (checkLatticeContainment(x, y, z) != -1);
            overlaps = checkLatticeOverlap(x, y, z, r);

            if(overlaps) {
              invalid = true;
            }

#ifdef EXTRACELLULAR
            if(contained)
              invalid = true;
#endif
#ifdef INTRACELLULAR
            if(! contained)
              invalid = true;
#endif
            // re-throw if the nanoparticle overlaps with another nanoparticle
            if(checkMNPOverlap(mnps, x, y, z, r))
              invalid = true;
        }

// Add in the thickness of the lipid layer surrounding intracellular MNP's
#ifdef LIPID_ENVELOPE
        if(contained)
          r += lipid_width;
#endif

        mnps->emplace_back(x, y, z, r, mmoment);
    }

#ifdef DEBUG_MNPS
    std::ofstream out_file;
    out_file.open("T2_sim_MNPs_unclustered.csv");
    out_file << "x,y,z,r,M" << std::endl;
    std::vector<MNP_info>::iterator i;
    for (i = mnps->begin(); i < mnps->end(); i++)
    {
        out_file << i->x << "," << i->y << "," << i->z << "," << i->r << ",";
        out_file << i->M << std::endl;
    }
    out_file.close();
#endif /* DEBUG_MNPS */

    print_mnp_stats(mnps);
    apply_bcs_on_mnps(mnps);
    return mnps;
}

// Handle the clustered MNP case
#elif defined CLUSTERED
/*
 * Initializes the magnetic nanoparticle clusters in different cells of the
 * lattice. There is a 50% chance a cell has no MNPs, and an equal chance that
 * it is any of the other types. Constrained such that the final [Fe] is the
 * same as the unclustered model.
 */
std::vector<MNP_info> *FCC::init_mnps(XORShift<> &gen)
{
    int numContained = 0;
    /* Magnetic moments established by HD on NV adjusted using SQUID magneto-
     * metry for saturation magnetization. Each Cell is an array of the magnetic
     * moments of the enclosed nanoparticles.*/
    std::vector<double> cells[CELL_TYPES];
    cells[0] = {5.35e-14/3.33, 1.688e-13/3.33};
    cells[1] = {1.72e-13/3.33, 8.50e-13/3.33, 1.200e-12/3.33, 2.6e-14/3.33,\
        1.7e-14/3.33, 1.7e-14/3.33, 7e-15/3.33, 1.9e-14/3.33};
    cells[2] = {1.76e-14/3.33, 2.3e-15/3.33, 1.5e-15/3.33, 1.531e-13/3.33};
    cells[3] = {1.126e-15*3.33, 4.68e-16*3.33};
    cells[4] = {2.129e-15*3.33, 3.653e-15*3.33};
    cells[5] = {2.096e-15*3.33,1.68e-15*3.33,2.3e-16*3.33};
    //cells[6] = {0.3356e-11/3.33, 0.0617e-11/3.33, 0.1249e-11/3.33, 0.0197e-11/3.33};
    std::vector<MNP_info> *mnps = new std::vector<MNP_info>;

// Assign MNPs to cells based on the cell library

    std::uniform_real_distribution<> dist(1 - (1/prob_labeled), 1);
    for (int i = 0; i < num_cells; i++)
    {
        double coin = dist(gen);

        /* Give each different MNP distribution an equal chance of occupying
         * a given cell. There is a preset chance that the cell is labeled at
         * all. */
        for (int j = CELL_TYPES - 1; j >= 0; j--)
        {
            if (coin > (double)j / (double)CELL_TYPES)
            {
                for (unsigned k = 0; k < cells[j].size(); k++)
                {
                    double M = cells[j][k];
                    double r = pow(mnp_pack*M/(1.6e-15), 1.0/3.0) * mnp_radius;
                    double x, y, z;
                    int containingCell;

                    /* Keep re-generating the center for the MNP in question
                     * until the MNP does not overlap with any other MNPs that
                     * have already been initialized. */
                    bool invalid = true;
                    while (invalid)
                    {
                        invalid = false;
                        double norm;

#ifdef INTRACELLULAR
                        norm = gen.rand_pos_double() * (cell_r);
#elif defined EXTRACELLULAR
                        norm = cell_r * (1 + gen.rand_pos_double()
                            * (u_throw_coeff - 1));
#elif defined INTRA_EXTRA
                        norm = gen.rand_pos_double() * cell_r * u_throw_coeff;
#endif
                        water_info loc = rand_displacement(norm, gen);

                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];

                        // Check if the MNP is contained by any cell
                        // other than the cell around which it is thrown -
                        // If so, re-throw.
                        containingCell = checkLatticeContainment(x, y, z);
                        if(containingCell != -1 && containingCell != i) {
                            invalid = true;
                        }

                        // Check for overlap with other MNPs, re-throw if so
                        if(checkMNPOverlap(mnps, x, y, z, r))
                            invalid = true;
                    } /* while(invalid) */

#ifdef LIPID_ENVELOPE
                    if(containingCell != -1) {
                        r += lipid_width;
                    }
#endif

                    /* Only actually place MNP in lattice if its center is
                     * inside the defined space (so we don't artificially
                     * increase MNP density via periodic boundary conditions) */
                    if (x  < bound && x > 0 && y < bound && y > 0 &&
                        z < bound && z > 0) {
                        mnps->emplace_back(x, y, z, r, M);

                        // For debugging purposes, count the # of MNPs in
                        // intracellular space
#ifdef DEBUG_MNPS
                        if(containingCell != -1)
                            numContained++;
#endif
                    }
                }
                break; // cell occupied -- don't try to fill it w/ more MNPs
            }
        }
    }

#ifdef DEBUG_MNPS
    std::ofstream out_file;
    std::cout << "Number of Intracellular MNPs: " << numContained << std::endl;
    out_file.open("T2_sim_MNPs_clustered.csv");
    out_file << "x,y,z,r,M" << std::endl;
    std::vector<MNP_info>::iterator i;
    for (i = mnps->begin(); i < mnps->end(); i++)
    {
        out_file << i->x << "," << i->y << "," << i->z << "," << i->r << ",";
        out_file << i->M << std::endl;
    }
    out_file.close();
#endif /* DEBUG_MNPS */

    print_mnp_stats(mnps);
    apply_bcs_on_mnps(mnps);
    return mnps;
}
#endif /* CLUSTERED */

/*
 * Prints out the number of nanoparticles in the lattice, the volume fraction of
 * iron, and the average radius of the nanoparticles initialized.
 */
void FCC::print_mnp_stats(std::vector<MNP_info> *mnps)
{
    double sum_V = 0, sum_r = 0;
    std::vector<MNP_info>::iterator np;
    for (np = mnps->begin(); np < mnps->end(); np++)
    {
        double rad = np->r;
        // Subtract away lipid envelope, if any, when calculating [Fe]
#ifdef LIPID_ENVELOPE
        if(checkLatticeContainment(np->x, np->y, np->z) != -1)
            rad -= lipid_width;
#endif

        sum_V += 4.0f/3.0f * pi * pow(rad, 3);
        sum_r += rad;
    }
#ifndef UNCLUSTERED
    sum_V /= mnp_pack;
#endif
    unsigned num_mnp = mnps->size();
    std::cout << "Volume fraction of MNPs: " << sum_V/pow(bound, 3);
    std::cout << std::endl << "Average MNP radius: " << sum_r/num_mnp;
    std::cout << "um" << std::endl << "Before applying boundary conditions, ";
    std::cout << "there were " << num_mnp << " nanoparticles." << std::endl;
}

/*
 * Determines the cell closest to the water molecule in question.
 */
void FCC::update_nearest_cell_full(water_info *w)
{
    /* Current position */
    double x = w->x;
    double y = w->y;
    double z = w->z;

    /* Determines distance from cell 0 to initialize data */
    int nearest = 0;
    double *center = fcc[0];
    double dx = x - center[0];
    double dy = y - center[1];
    double dz = z - center[2];
    double min_dist = NORMSQ(dx, dy, dz);

    /* Check distance to/from ALL cells */
    for (int i = 1; i < num_cells; i++)
    {
        center = fcc[i];
        dx = x - center[0];
        dy = y - center[1];
        dz = z - center[2];
        double curr_dist = NORMSQ(dx, dy, dz);
        if (curr_dist < min_dist)
        {
            min_dist = curr_dist;
            nearest = i;
        }
    }
    /* Update water molecule's record of nearest cell */
    w->nearest = nearest;
    if (in_cell(w))
        w->in_cell = true;
    else
        w->in_cell = false;
}

/*
 * For water molecules that have not crossed a boundary, this function updates
 * the molecule's information about the cell it is closest to.
 */
void FCC::update_nearest_cell(water_info *w)
{
    /* Current position */
    double x = w->x;
    double y = w->y;
    double z = w->z;

    /* Information about nearest cell center on the last timestep */
    int nearest = w->nearest;
    int *check = neighbors[nearest];
    double *center = fcc[nearest];
    double dx = x - center[0];
    double dy = y - center[1];
    double dz = z - center[2];
    double min_dist = NORMSQ(dx, dy, dz);

    /* Check distance to/from all neighboring cells */
    for (int i = 0; i < num_neighbors; i++)
    {
        int curr_check = check[i];
        double *curr_center = fcc[curr_check];
        dx = x - curr_center[0];
        dy = y - curr_center[1];
        dz = z - curr_center[2];

        double curr_dist = NORMSQ(dx, dy, dz);
        if (curr_dist < min_dist)
        {
            min_dist = curr_dist;
            nearest = curr_check;
        }
    }

    /* Update water molecule's record of nearest cell */
    w->nearest = nearest;
}

/*
 * Determines if a molecule still resides in the cell it is associated with
 */
bool FCC::in_cell(water_info *w)
{
    double *center = fcc[w->nearest];
    double x = w->x - center[0];
    double y = w->y - center[1];
    double z = w->z - center[2];
    return cell_r * cell_r > NORMSQ(x, y, z);
}

/*
 * Determines whether or not the water molecule has crossed a boundary. The
 * appropriate additions are applied if it has.
 */
bool FCC::boundary_conditions(water_info *w)
{
    bool cross = false;

    if (w->x > bound) {
        w->x -= bound;
        w->cross_x++;
        cross = true;
    } else if (w->x < 0) {
        w->x += bound;
        w->cross_x--;
        cross = true;
    }

    if (w->y > bound) {
        w->y -= bound;
        w->cross_y++;
        cross = true;
    } else if (w->y < 0) {
        w->y += bound;
        w->cross_y--;
        cross = true;
    }

    if (w->z > bound) {
        w->z -= bound;
        w->cross_z++;
        cross = true;
    } else if (w->z < 0) {
        w->z += bound;
        w->cross_z--;
        cross = true;
    }

    return cross;
}

/*
 * Simulates one step of diffusion for a single water molecule, w. Also
 * accounts for reflections off of magnetic nanoparticles. Returns the value
 * of the B field pre-hashed at the node the water molecule resides in.
 */
double FCC::diffusion_step(water_info *w, Octree *tree, XORShift<> &gen)
{
    water_info disp;
    int old_nearest = w->nearest;
    oct_node *voxel = tree->get_voxel(w);

    // check for reflection off a cell membrane if w starts inside a cell
    if (w->in_cell)
    {
        disp = rand_displacement(norm_in(gen), gen);
        *w += disp;
        update_nearest_cell(w);

        if (!in_cell(w))
        {
            if (gen.rand_pos_double() < reflectIO)
            {
                *w -= disp;
                w->nearest = old_nearest;
                return tree->get_field(w, voxel);
            }
            else
                w->in_cell = false;
        }
    }

    // check for reflection off a cell membrane if w starts outside a cell
    else
    {
        disp = rand_displacement(norm_out(gen), gen);
        *w += disp;
        update_nearest_cell(w);

        if (in_cell(w))
        {
            if (gen.rand_pos_double() < reflectOI)
            {
                *w -= disp;
                w->nearest = old_nearest;
                return tree->get_field(w, voxel);
            }
            else
                w->in_cell = true;
        }
    }

    // check for reflection off a magnetic nanoparticle
    std::vector<MNP_info> *mnps = voxel->resident;
    if (mnps)
    {
        std::vector<MNP_info>::iterator i;
        for (i = mnps->begin(); i < mnps->end(); i++)
        {
            double dx = w->x - i->x;
            double dy = w->y - i->y;
            double dz = w->z - i->z;
            double r = i->r;
            if (NORMSQ(dx, dy, dz) < r * r)
            {
                *w -= disp;
                w->nearest = old_nearest;
                if (in_cell(w))
                    w->in_cell = true;
                else
                    w->in_cell = false;
                return tree->get_field(w, voxel);
            }
        }
    }

    // Account for periodic boundary conditions
    if (boundary_conditions(w))
        update_nearest_cell_full(w);

    return tree->get_field(w);
}

#ifdef FULL_BOUNDARIES
/*
 * Duplicates existing nanoparticles initialized in the lattice across all
 * periodic boundaries.
 */
void FCC::apply_bcs_on_mnps(std::vector<MNP_info> *mnps)
{
    unsigned init_size = mnps->size();
    for (unsigned i = 0; i < init_size; i++)
    {
        MNP_info *np = mnps->data() + i;
        double x = np->x, y = np->y, z = np->z, r = np->r, M = np->M;

        // boundaries touching faces
        mnps->emplace_back(x - bound, y, z, r, M);
        mnps->emplace_back(x - bound, y, z, r, M);
        mnps->emplace_back(x, y - bound, z, r, M);
        mnps->emplace_back(x, y + bound, z, r, M);
        mnps->emplace_back(x, y, z - bound, r, M);
        mnps->emplace_back(x, y, z + bound, r, M);

        // boundaries touching edges
        mnps->emplace_back(x, y - bound, z - bound, r, M);
        mnps->emplace_back(x, y - bound, z + bound, r, M);
        mnps->emplace_back(x, y + bound, z - bound, r, M);
        mnps->emplace_back(x, y + bound, z + bound, r, M);

        mnps->emplace_back(x - bound, y, z - bound, r, M);
        mnps->emplace_back(x - bound, y, z + bound, r, M);
        mnps->emplace_back(x + bound, y, z - bound, r, M);
        mnps->emplace_back(x + bound, y, z + bound, r, M);

        mnps->emplace_back(x - bound, y - bound, z, r, M);
        mnps->emplace_back(x - bound, y + bound, z, r, M);
        mnps->emplace_back(x + bound, y - bound, z, r, M);
        mnps->emplace_back(x + bound, y + bound, z, r, M);

        // boundaries touching corners
        mnps->emplace_back(x - bound, y - bound, z - bound, r, M);
        mnps->emplace_back(x - bound, y - bound, z + bound, r, M);
        mnps->emplace_back(x - bound, y + bound, z - bound, r, M);
        mnps->emplace_back(x - bound, y + bound, z + bound, r, M);
        mnps->emplace_back(x + bound, y - bound, z - bound, r, M);
        mnps->emplace_back(x + bound, y - bound, z + bound, r, M);
        mnps->emplace_back(x + bound, y + bound, z - bound, r, M);
        mnps->emplace_back(x + bound, y + bound, z + bound, r, M);
    }
}

#else
/*
 * Of the nanoparticles initially initialized, if some are within a pre-set
 * range of the border of the space being simulated, those nanoparticles will
 * be duplicated across the appropriate periodic boundaries.
 */
void FCC::apply_bcs_on_mnps(std::vector<MNP_info> *mnps)
{
    unsigned init_size = mnps->size();
    for (unsigned i = 0; i < init_size; i++)
    {
        MNP_info *np = mnps->data() + i;
        double x = np->x, y = np->y, z = np->z, r = np->r, M = np->M;

        if (x + border > bound) // near front side
        {
            mnps->emplace_back(x - bound, y, z, r, M);

            if (y + border > bound) // near front and right sides
            {
                mnps->emplace_back(x - bound, y - bound, z, r, M);
                mnps->emplace_back(x, y - bound, z, r, M);

                if (z + border > bound) // near front, right, and top sides
                {
                    mnps->emplace_back(x - bound, y - bound, z - bound, r, M);
                    mnps->emplace_back(x - bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y - bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near front, right, and bottom sides
                {
                    mnps->emplace_back(x - bound, y - bound, z + bound, r, M);
                    mnps->emplace_back(x - bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y - bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else if (y - border < 0) // near front and left sides
            {
                mnps->emplace_back(x - bound, y + bound, z, r, M);
                mnps->emplace_back(x, y + bound, z, r, M);

                if (z + border > bound) // near front, left, and top sides
                {
                    mnps->emplace_back(x - bound, y + bound, z - bound, r, M);
                    mnps->emplace_back(x - bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y + bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near front, left, and bottom sides
                {
                    mnps->emplace_back(x - bound, y + bound, z + bound, r, M);
                    mnps->emplace_back(x - bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y + bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else // not near left or right sides, but near front side
            {
                if (z + border > bound) // near front and top sides
                {
                    mnps->emplace_back(x - bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }
                else if (z - border < 0) // near front and bottom sides
                {
                    mnps->emplace_back(x - bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }
        }

        else if (x - border < 0) // near back side
        {
            mnps->emplace_back(x + bound, y, z, r, M);

            if (y + border > bound) // near back and right sides
            {
                mnps->emplace_back(x + bound, y - bound, z, r, M);
                mnps->emplace_back(x, y - bound, z, r, M);

                if (z + border > bound) // near back, right, and top sides
                {
                    mnps->emplace_back(x + bound, y - bound, z - bound, r, M);
                    mnps->emplace_back(x + bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y - bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near back, right, and bottom sides
                {
                    mnps->emplace_back(x + bound, y - bound, z + bound, r, M);
                    mnps->emplace_back(x + bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y - bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else if (y - border < 0) // near back and left sides
            {
                mnps->emplace_back(x + bound, y + bound, z, r, M);
                mnps->emplace_back(x, y + bound, z, r, M);

                if (z + border > bound) // near back, left, and top sides
                {
                    mnps->emplace_back(x + bound, y + bound, z - bound, r, M);
                    mnps->emplace_back(x + bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y + bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near back, left, and bottom sides
                {
                    mnps->emplace_back(x + bound, y + bound, z + bound, r, M);
                    mnps->emplace_back(x + bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y + bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else // not near left or right sides, but near back side
            {
                if (z + border > bound) // near back and top sides
                {
                    mnps->emplace_back(x + bound, y, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }
                else if (z - border < 0) // near back and bottom sides
                {
                    mnps->emplace_back(x + bound, y, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }
        }

        else // not near back or front sides
        {
            if (y + border > bound) // near right side
            {
                mnps->emplace_back(x, y - bound, z, r, M);

                if (z + border > bound) // near top and right sides
                {
                    mnps->emplace_back(x, y - bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near bottom and right sides
                {
                    mnps->emplace_back(x, y - bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else if (y - border < 0) // near left side
            {
                mnps->emplace_back(x, y + bound, z, r, M);

                if (z + border > bound) // near top and left sides
                {
                    mnps->emplace_back(x, y + bound, z - bound, r, M);
                    mnps->emplace_back(x, y, z - bound, r, M);
                }

                else if (z - border < 0) // near bottom and left sides
                {
                    mnps->emplace_back(x, y + bound, z + bound, r, M);
                    mnps->emplace_back(x, y, z + bound, r, M);
                }
            }

            else // not near left, right, front, or back sides
            {
                if (z + border > bound) // near top side
                    mnps->emplace_back(x, y, z - bound, r, M);
                else if (z - border < 0) // near bottom side
                    mnps->emplace_back(x, y, z + bound, r, M);
            }
        }
    }
}
#endif
