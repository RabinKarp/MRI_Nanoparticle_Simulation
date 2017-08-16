/*
 * @author  Aadyot Bhatnagar
 * @date    August 12, 2016
 * @file    fcc_diffusion.cpp
 * @brief   Implementation details for a class that represents a 3x3x3 face-
 *          centered cubic lattice of cells with periodic boundary conditions.
 *          The only public method is one that simulates one step of the
 *          diffusion of a water molecule through the cell lattice.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include "parameters.h"
#include "fcc_diffusion.h"
#include "rand_walk.h"

#define CELL_TYPES 6

using namespace std;

Triple* FCC::linearLattice() {
    Triple *linLattice = new Triple[num_cells];

    for(int i = 0; i < num_cells; i++) {
        linLattice[i].x = fcc[i][0];
        linLattice[i].y = fcc[i][1];
        linLattice[i].z = fcc[i][2];
    }
    return linLattice;
}

void FCC::initializeLookupTable() {
    lookupTable = new int*[hashDim * hashDim * hashDim];
    double cubeLength = bound / hashDim;
    double diagonal = sqrt(3) * cubeLength;

    for(int i = 0; i < hashDim * hashDim * hashDim; i++) {
        vector<int> cells;
        double x = (i % hashDim) * cubeLength;
        double y = ((i / hashDim) % (hashDim)) * cubeLength;
        double z = i / (hashDim * hashDim) * cubeLength;

        for(int j = 0; j < num_cells; j++) {
            double dx = fcc[j][0] - x;
            double dy = fcc[j][1] - y;
            double dz = fcc[j][2] - z;
            if(sqrt(NORMSQ(dx, dy, dz)) < cell_r + diagonal) {
                cells.push_back(j);
            }
        }

        assert(cells.size() < maxNeighbors);

        lookupTable[i] = new int[maxNeighbors];

        for(int j = 0; j < maxNeighbors; j++) {
            if(j < cells.size())
                lookupTable[i][j] = cells[j];
            else
                lookupTable[i][j] = -1;
        }
    }
}

/*
 * Initializes the FCC lattice.
 */
FCC::FCC(double D_in, double D_out, double P_expr, XORShift<> &gen)
{
    norm_in = std::normal_distribution<>(0, sqrt(pi * D_in * tau));
    norm_out = std::normal_distribution<>(0, sqrt(pi * D_out * tau));
    this->reflectIO = 1 - sqrt(tau / (6*D_in)) * 4 * P_expr;
    this->reflectOI = 1 - ((1 - reflectIO) * sqrt(D_in/D_out));
    initializeCells(gen);
    initializeLookupTable();
}

/**
 * Destroy the lookup the table for the face-centered cubic lattice.
 */
FCC::~FCC() {
    for(int i = 0; i < hashDim * hashDim * hashDim; i++) {
        if(lookupTable[i])
            delete[] lookupTable[i];
    }
}
/**
 * Initializes cells within the simulation bound by randomly throwing them
 * and checking that they don't overlap with each other. If cells do overlap,
 * they are simply re-thrown.
 */
void FCC::initializeCells(XORShift<> &gen) {
    for(int i = 0; i < num_cells; i++) {
        bool invalid = true;
        double x, y, z;
        while(invalid) {
            invalid = false;
            double x = cell_r + gen.rand_pos_double() * (bound - 2 * cell_r);
            double y = cell_r + gen.rand_pos_double() * (bound - 2 * cell_r);
            double z = cell_r + gen.rand_pos_double() * (bound - 2 * cell_r);

            // Check against overlap with other cells
            for(int j = 0; j < i; j++) {
                double dx = fcc[j][0] - x;
                double dy = fcc[j][1] - y;
                double dz = fcc[j][2] - z;

                if(NORMSQ(dx, dy, dz) < 4 * cell_r * cell_r)
                    invalid = true;
            }
        }
        fcc[i][0] = x;
        fcc[i][1] = y;
        fcc[i][2] = z;
    }
}

/*
 * Given a set of Cartesian boundary conditions, initialize a given number of
 * water molecules.
 */
water_info *FCC::init_molecules(int n, XORShift<> &gen)
{
    water_info *molecules = new water_info[n];
    water_info *temp = molecules;

    double offset = (bound - water_start_bound) / 2.0;

    for (int i = 0; i < n; i++)
    {
        double x = offset + gen.rand_pos_double() * water_start_bound;
        double y = offset + gen.rand_pos_double() * water_start_bound;
        double z = offset + gen.rand_pos_double() * water_start_bound;

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
           && sqDist > pow((cell_r - r), 2)) {
           overlaps = true;
       }
   }
   return overlaps;
}

/**
 * Initialize an MNP (a dipole for the cell) at the center of each cell with
 * the specified magnetic moment.
 */
std::vector<MNP_info> *FCC::init_mnps() {
    vector<MNP_info> *mnps = new vector<MNP_info>();
    for(int i = 0; i < num_cells; i++) {
        mnps->emplace_back(fcc[i][0], fcc[i][1], fcc[i][2], 0, mmoment);
    }
    return mnps;
}

/**
* Helper function that checks if the given (x, y, z) coordinate is
* contained in any cell of the FCC lattice. Returns the index of the lattice
* cell that contains the given point, or -1 if no cell contains that point.
*/
int FCC::checkLatticeContainment(double x, double y, double z) {
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

#ifdef UNCLUSTERED
/*
 * Initializes the specified number of individaul, unclustered magnetic
 * nanoparticles all in extracellular space.
 */
std::vector<MNP_info> *FCC::init_mnps(XORShift<> &gen)
{
    std::vector<MNP_info> *mnps = new std::vector<MNP_info>;
    for (int i = 0; i < num_mnps; i++)
    {
        double x, y, z;
        bool invalid = true;

        // keep generating (x,y,z) coordinates until we get an extracellular one
        // that doesn't overlap with other nanoparticles
        while (invalid)
        {
            x = gen.rand_pos_double() * bound;
            y = gen.rand_pos_double() * bound;
            z = gen.rand_pos_double() * bound;
            invalid = false;

            // re-throw if the nanoparticle is inside/outside a cell (depends on flag)
            for (int j = 0; j < num_cells && !invalid; j++)
            {
                double dx = x - fcc[j][0];
                double dy = y - fcc[j][1];
                double dz = z - fcc[j][2];

                // Check for invalid MNP's (overlapping with cell boundaries,
                // not in desired intracellular or extracellular locations, and
                // set a flag to rethrow if needed)
#if defined EXTRACELLULAR
                if (NORMSQ(dx, dy, dz) < pow(cell_r + mnp_radius, 2)) {
                    invalid = true;
                }
#elif defined INTRACELLULAR
                if ((checkLatticeContainment(x, y, z) == -1) ||
                    checkLatticeOverlap(x, y, z, mnp_radius)) {
                    invalid = true;
                }
// Just check for cell boundary overlap in this case
#elif defined INTRA_EXTRA
                if ((NORMSQ(dx, dy, dz) > pow(cell_r - mnp_radius, 2))
                    && (NORMSQ(dx, dy, dz) < pow(cell_r + mnp_radius, 2)))
                    invalid = true;
#endif
            }

            // re-throw if the nanoparticle overlaps with another nanoparticle
            std::vector<MNP_info>::iterator curr;
            for (curr = mnps->begin(); curr != mnps->end() && !invalid; curr++)
            {
                double dx = x - curr->x;
                double dy = y - curr->y;
                double dz = z - curr->z;
                if (NORMSQ(dx, dy, dz) < pow(2*mnp_radius, 2))
                    invalid = true;
            }
        }

        // If the flag is set, factor in the new lipid envelope
        double radius = mnp_radius;
#ifdef LIPID_ENVELOPE
        if(checkLatticeContainment(x, y, z) != -1)
            radius += lipid_width;
#endif
        mnps->emplace_back(x, y, z, radius, mmoment);
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
/* endif UNCLUSTERED */
#elif defined CLUSTERED
/*
 * Initializes the magnetic nanoparticle clusters in different cells of the
 * lattice. There is a 50% chance a cell has no MNPs, and an equal chance that
 * it is any of the other types. Constrained such that the final [Fe] is the
 * same as the extracellular model.
 */
std::vector<MNP_info> *FCC::init_mnps(XORShift<> &gen)
{
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

                    /* Keep re-generating the center for the MNP in question
                     * until the MNP does not overlap with any other MNPs that
                     * have already been initialized. */
                    bool invalid = true;
                    while (invalid)
                    {
                        invalid = false;
#ifdef INTRACELLULAR
                        double norm = gen.rand_pos_double() * (cell_r);
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];
#elif defined EXTRACELLULAR
#ifdef THROW_FREE // Throw cluster anywhere in extracellular space - the check
                  // against cell containment occurs after the branch
                        x = gen.rand_pos_double() * bound;
                        y = gen.rand_pos_double() * bound;
                        z = gen.rand_pos_double() * bound;

#else // Throw the particle within a certain vicinity of the labeled cell
                        double norm = cell_r * (1 + gen.rand_pos_double()
                            * (u_throw_coeff - 1));
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];
#endif
                        // Re-throw in case of cell containment, for both cases
                        // THROW_FREE set and unset
                        if(checkLatticeContainment(x, y, z) != -1)
                            invalid = true;
#elif defined INTRA_EXTRA
                        double norm = gen.rand_pos_double() * cell_r * u_throw_coeff;
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];

                        // If the MNP center occurs inside ANOTHER cell,
                        // check and re-throw
                        if(checkLatticeContainment(x, y, z) != -1 &&
                            checkLatticeContainment(x, y, z) != i)
                            invalid = true;
#endif
                        std::vector<MNP_info>::iterator m, start = mnps->begin();
                        for (m = start; m != mnps->end() && !invalid; m++)
                        {
                            double dx = x - m->x;
                            double dy = y - m->y;
                            double dz = z - m->z;
                            if (NORMSQ(dx, dy, dz) < pow(r + m->r, 2))
                                invalid = true;
                        }
                    }

                    /* Only actually place MNP in lattice if its center is
                     * inside the defined space (so we don't artificially
                     * increase MNP density via periodic boundary conditions) */
                    if (x  < bound && x > 0 && y < bound && y > 0 &&
                        z < bound && z > 0) {
#ifdef LIPID_ENVELOPE
                        if(checkLatticeContainment(x, y, z) != -1)
                            r += lipid_width;
#endif
                        mnps->emplace_back(x, y, z, r, M);
                    }
                }
                break; // cell occupied -- don't try to fill it w/ more MNPs
            }
        }
    }
#ifdef DEBUG_MNPS
    std::ofstream out_file;
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

    /* Check distance to/from all neighboring cells */
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

    double cubeLength = bound / hashDim;
    int x_idx = x / cubeLength;
    int y_idx = y / cubeLength;
    int z_idx = z / cubeLength;

    int* nearest =
        lookupTable[z_idx * hashDim * hashDim
            + y_idx * hashDim
            + x_idx];

    double cDist = bound * sqrt(3);

    int cIndex = -1;
    while(*nearest != -1) {
        double dx = fcc[*nearest][0] - x;
        double dy = fcc[*nearest][1] - y;
        double dz = fcc[*nearest][2] - z;

        double dist = NORMSQ(dx, dy, dz);
        if(NORMSQ(dx, dy, dz) < cDist) {
            cDist = dist;
            cIndex = *nearest;
        }
        nearest++;
    }

    w->nearest = cIndex;
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
