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
#include "parameters.h"
#include "fcc_diffusion.h"
#include "rand_walk.h"

#define CELL_TYPES 6

/*
 * Uses cell radius to scale coordinates of cell centers in FCC lattice and
 * initializes the scattering probabilities upon collision with a cell membrane,
 * both from the inside and from the outside. Also intializes the standard
 * deviation expected from the Gaussian distribution used to generate random
 * displacement vectors.
 */
FCC::FCC(double D_in, double D_out, double P_expr)
{
    for (int i = 0; i < 172; i++)
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
 * in a given vector. The target MNP
 */
inline bool FCC::checkOverlap(std::vector<MNP_info> *mnps,
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
        bool contained;
        bool invalid = true;

        // keep generating (x,y,z) coordinates until we get an extracellular one
        // that doesn't overlap with other nanoparticles
        while (invalid)
        {
            x = gen.rand_pos_double() * bound;
            y = gen.rand_pos_double() * bound;
            z = gen.rand_pos_double() * bound;
            invalid = false;
            contained = false;

            // Check for cell containment and cell boundary overlap
            for (int j = 0; j < num_cells && !invalid; j++)
            {
                double dx = x - fcc[j][0];
                double dy = y - fcc[j][1];
                double dz = z - fcc[j][2];

                double sqDist = NORMSQ(dx, dy, dz);

#ifdef EXTRACELLULAR
                if (sqDist < pow(cell_r + mnp_radius, 2))
                    invalid = true;
#endif
#ifdef INTRACELLULAR
                contained = true;
                if (sqDist > pow(max(0, cell_r - mnp_radius), 2))
                    invalid = true;
#endif

// In this case, check for cell boundary overlap and test for containment in
// any cell.
#ifdef INTRA_EXTRA
                if (sqDist < pow(cell_r + mnp_radius, 2)) {
                  contained = true;
                  if(sqDist > pow(max(0, cell_r - mnp_radius), 2)) {
                    invalid = true;
                  }
                }
#endif
            }
            // re-throw if the nanoparticle overlaps with another nanoparticle
            if(checkOverlap(mnps, x, y, z, r))
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
#else
/*
 * Initializes the magnetic nanoparticle clusters in different cells of the
 * lattice. There is a 50% chance a cell has no MNPs, and an equal chance that
 * it is any of the other types. Constrained such that the final [Fe] is the
 * same as the unclustered model.
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

// Assign MNPs to cells based on the cell library
#ifdef INTRACELLULAR

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

#ifdef LIPID_ENVELOPE
                    r += lipid_width;
#endif

                    double x, y, z;

                    /* Keep re-generating the center for the MNP in question
                     * until the MNP does not overlap with any other MNPs that
                     * have already been initialized. */
                    bool invalid = true;
                    while (invalid)
                    {
                        invalid = false;
                        double norm = gen.rand_pos_double() * cell_r;
                        water_info loc = rand_displacement(norm, gen);
                        x = loc.x + fcc[i][0];
                        y = loc.y + fcc[i][1];
                        z = loc.z + fcc[i][2];

                        if(checkOverlap(mnps, x, y, z, r))
                          invalid = true;
                    }

                    /* Only actually place MNP in lattice if its center is
                     * inside the defined space (so we don't artificially
                     * increase MNP density via periodic boundary conditions) */
                    if (x  < bound && x > 0 && y < bound && y > 0 &&
                        z < bound && z > 0)
                        mnps->emplace_back(x, y, z, r, M);
                }
                break; // cell occupied -- don't try to fill it w/ more MNPs
            }
        }
    }
#endif /* INTRACELLULAR */

/*
 * Throw extracellular clustered MNPs by randomly throwing clusters with
 * different magnetic moments and diffusion boundary radii. In this
 * implementation, each cluster has an equal chance of being thrown.
 */
#ifdef EXTRACELLULAR
  int numMoments = 21;
  double momentList[] = {
    5.35e-14/3.33, 1.688e-13/3.33, 1.72e-13/3.33,
    8.50e-13/3.33, 1.200e-12/3.33, 2.6e-14/3.33,
    1.7e-14/3.33, 1.7e-14/3.33, 7e-15/3.33,
    1.9e-14/3.33, 1.76e-14/3.33, 2.3e-15/3.33,
    1.5e-15/3.33, 1.531e-13/3.33, 1.126e-15*3.33,
    4.68e-16*3.33, 2.129e-15*3.33, 3.653e-15*3.33,
    2.096e-15*3.33,1.68e-15*3.33, 2.3e-16*3.33
  };
  std::uniform_real_distribution<> dist(0, 21);

  for(int i = 0; i < num_mnps; i++) {

    int coin = (int) dist(gen);
    double M = momentList[coin];
    double r = pow(mnp_pack*M/(1.6e-15), 1.0/3.0) * mnp_radius;
    bool invalid = false;

    while(invalid) {
      x = gen.rand_pos_double() * bound;
      y = gen.rand_pos_double() * bound;
      z = gen.rand_pos_double() * bound;

      // Check against cell containment
      for (int j = 0; j < num_cells && !invalid; j++)
      {
          double dx = x - fcc[j][0];
          double dy = y - fcc[j][1];
          double dz = z - fcc[j][2];

          double sqDist = NORMSQ(dx, dy, dz);

          if (sqDist < pow(cell_r + r, 2))
              invalid = true;
      }

      // Check MNP overlap
      if(checkOverlap(mnps, x, y, z, r))
          invalid = true;
    }
    mnps->emplace_back(x, y, z, r, mmoment);
  }

#endif

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
 * Prints out the number of nanoparticles in the lattice, the volume fraction of
 * iron, and the average radius of the nanoparticles initialized.
 */
void FCC::print_mnp_stats(std::vector<MNP_info> *mnps)
{
    double sum_V = 0, sum_r = 0;
    std::vector<MNP_info>::iterator np;
    for (np = mnps->begin(); np < mnps->end(); np++)
    {
        sum_V += 4.0f/3.0f * pi * pow(np->r, 3);
        sum_r += np->r;
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
