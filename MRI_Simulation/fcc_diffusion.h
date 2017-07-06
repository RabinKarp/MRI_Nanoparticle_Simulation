/*
 * @author  Aadyot Bhatnagar
 * @date    August 10, 2016
 * @file    fcc_diffusion.h
 * @brief   Header file outlining a class that represents a 3x3x3 face-centered
 *          cubic lattice of cells. This file contains most of the parameters
 *          that affect the diffusion behavior and specific calculations used
 *          in the T2 simulation.
 */

#ifndef FCC_DIFFUSION_H
#define FCC_DIFFUSION_H

#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include "rand_walk.h"
#include "octree.h"

/* Intrinsic properties of a face-centered cubic lattice of spheres */
const int num_cells = 172;          // number of cells in the FCC lattice
const int num_neighbors = 12;       // number of neighbors each FCC cell has

/*
 * This class encodes a 3x3x3 face-centered cubic lattice with periodic
 * boundary conditions. These cell boundaries are used to initialize magnetic
 * nanoparticles and water molecules throughout the lattice.
 */
class FCC
{
    public:
    FCC(double D_in, double D_out, double P);
    double diffusion_step(water_info *w, Octree *tree, XORShift<> &gen);
    std::vector<MNP_info> *init_mnps(XORShift<> &gen);
    water_info *init_molecules(double L, int n, std::vector<MNP_info> *mnps,\
        XORShift<> &gen);
    void update_nearest_cell_full(water_info *w);

    private:
    double reflectIO, reflectOI;
    std::normal_distribution<> norm_in, norm_out;

    bool in_cell(water_info *w);
    bool boundary_conditions(water_info *w);
    void update_nearest_cell(water_info *w);
    void print_mnp_stats(std::vector<MNP_info> *mnps);
    void apply_bcs_on_mnps(std::vector<MNP_info> *mnps);

    /*
     * Instance variable representing the centers of all the cells in an FCC
     * lattice (unscaled).
     */
    double fcc[num_cells][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},\
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

    /*
     * Instance variable where the array stored at the ith index corresponds to
     * the list of all the nieghbors of the ith cell in the fcc array above.
     */
    int neighbors[num_cells][num_neighbors] = {{53,56,58,9,8,7,55,5,4,2,1,6},\
    {   0,   40,   41,   32,   11,   10,   48,    6,    5,    3,    2,    7},\
    {   0,   37,   39,   30,   10,    8,   13,    6,   46,    3,    4,    1},\
    {   1,    2,    4,    5,   22,   20,   10,   11,   12,   13,   19,   17},\
    {   0,   43,   35,   27,   13,   12,    9,   34,    8,    5,    2,    3},\
    {   0,    1,    3,    4,   45,    7,   29,    9,   38,   11,   12,   36},\
    {   0,   39,   41,   46,   48,   49,   58,   59,   61,   56,    1,    2},\
    {   0,    1,   40,   45,   47,    5,   48,   55,   57,   58,   60,   38},\
    {   0,   56,   54,   51,   46,   44,   43,   37,   35,   53,    2,    4},\
    {   0,   42,   43,   45,   50,   52,   53,   55,   36,   34,    4,    5},\
    {  23,    1,    2,    3,   32,   39,   20,   30,   25,   33,   22,   41},\
    {  21,   19,   38,   29,   22,   31,   24,    5,   32,    3,   40,    1},\
    {  17,   29,   27,   26,   19,   36,   16,   14,   34,    5,    4,    3},\
    {  17,   30,   28,   27,   20,   18,   37,   15,   35,    2,    3,    4},\
    {  12,   17,   16,   27,  121,  117,  115,   26,  158,   79,   71,   69},\
    { 119,  123,   18,   17,   81,   13,   73,   71,  117,   27,   28,  162},\
    {  29,   75,  115,  121,  125,   65,   26,   19,   14,   12,   63,  150},\
    {  14,   19,   15,   27,   13,   12,   20,  117,  121,  123,    3,  127},\
    {  64,  123,  119,  151,   74,   62,   30,   28,   20,   15,   13,  129},\
    {  21,   17,   16,   12,   22,   29,  131,   11,  121,  127,    3,  125},\
    { 129,  123,   23,   22,  127,   18,   17,   13,   10,   30,    3,  133},\
    {  29,  135,   19,   65,  131,   31,  125,  154,   11,   77,   67,   24},\
    {  20,   19,  127,   24,   11,   10,  131,   32,   25,  133,  137,    3},\
    {  64,   30,   20,   33,   25,  133,   10,  155,  139,   76,  129,   66},\
    { 135,   32,   68,   31,  131,   70,   21,   22,  159,  137,   11,   78},\
    { 137,   72,  163,   33,   32,  139,   80,   23,   10,   22,   70,  133},\
    {  79,   16,   14,   69,   12,  144,   63,   89,   83,   34,   75,   36},\
    {  79,   81,   35,   34,    4,   71,   17,   91,   12,   13,   14,   15},\
    {  81,   18,   15,   13,   82,  145,   35,   37,   93,   73,   62,   74},\
    {  85,   38,   36,   65,   75,    5,   77,   21,   11,   12,   19,   16},\
    {  37,   76,   64,   84,   39,   23,   20,   18,   13,   74,    2,   10},\
    {  38,   77,   68,   24,   11,   67,  146,   21,   88,   87,   78,   40},\
    {  78,    1,   80,   24,   70,   22,   90,   10,   11,   40,   25,   41},\
    {  76,   39,   72,  147,   80,   25,   23,   10,   41,   92,   86,   66},\
    {  89,   36,   27,   26,   79,   43,   91,   99,   12,   42,    4,    9},\
    {  28,   27,  101,    4,   37,   43,   93,    8,   91,   44,   81,   13},\
    {  85,   26,   42,   45,   75,   29,   83,   12,   95,   34,    5,    9},\
    {  13,   30,   35,   74,   94,   44,   46,   84,   28,    8,    2,   82},\
    {  85,   29,   40,   45,   47,   87,   11,   97,   77,   31,    7,    5},\
    {  33,   41,   46,   49,   96,   10,   30,    6,    2,   76,   84,   86},\
    {  78,    1,   38,   47,   48,   32,   31,    7,   11,   88,   98,   90},\
    {  39,    1,   33,   80,    6,   90,   92,   10,   48,   49,  100,   32},\
    { 103,   50,   52,   83,   89,   95,   99,   36,  109,   34,    9,  140},\
    {  34,   53,   50,  111,    9,   35,    8,   99,    4,   91,  101,   51},\
    { 113,   82,   54,  141,   51,    8,   35,   94,  102,   37,  101,   93},\
    {  85,   38,   52,    9,   36,    7,   55,    5,   95,   97,  105,   57},\
    {  39,   37,   54,   56,   59,    8,  104,   96,   94,    2,   84,    6},\
    {  98,   57,   60,   87,   88,   97,   40,  107,  108,   38,    7,  142},\
    {  98,    1,   40,   41,   90,  110,    6,    7,  100,   61,   60,   58},\
    { 100,   41,   59,   61,   86,   92,   96,  106,  112,   39,  143,    6},\
    {  52,   53,  156,   43,   42,   99,    9,  109,  111,  114,  116,  120},\
    { 116,   53,   43,  101,   44,   54,  111,  113,    8,  118,  160,  122},\
    { 105,  103,  114,   55,  120,  124,    9,   50,  148,   45,   42,   95},\
    {   0,   56,   55,   51,   50,   43,  116,  120,  126,  122,    8,    9},\
    { 128,  122,  118,   94,  149,  102,  104,    8,   44,   56,   46,   51},\
    {   0,   58,   57,  120,  124,  126,  130,   52,   53,    7,   45,    9},\
    {   0,  122,  132,   46,  128,   54,   58,   53,   59,    8,  126,    6},\
    { 105,   45,  152,   47,   55,  134,    7,  130,  124,   60,  107,   97},\
    {   0,  126,  136,   55,   56,   60,  132,   61,  130,   48,    7,    6},\
    { 104,   46,  153,   49,  138,    6,   56,  132,  128,   61,  106,   96},\
    {  48,   47,  108,  157,  110,   98,   57,    7,  130,   58,  134,  136},\
    { 161,   48,   49,  138,    6,  136,  132,   58,   59,  112,  110,  100},\
    { 119,   74,  165,   73,  151,  145,   18,   28,   64,   82,   15,   30},\
    {  16,  115,  150,  144,   75,   69,  164,   26,   83,   14,   65,  158},\
    {  18,  155,   23,   74,   76,  129,   30,  151,   66,   20,   84,   62},\
    {  75,   77,  125,   29,   21,   16,  150,  154,   85,   63,   19,   67},\
    { 155,   23,  147,   76,  139,   33,  167,   72,   86,   25,   64,   92},\
    {  31,  146,  135,  154,   21,   77,   68,  166,   87,   65,   24,  159},\
    {  31,   24,   67,  135,  159,  146,  166,   78,   88,   70,   21,  131},\
    {  63,   14,  115,  144,  158,   79,  164,   26,   89,   16,   71,   75},\
    {  80,   25,  159,  163,   32,   24,  137,   78,   68,   90,   72,   22},\
    {  81,   27,  158,  117,  162,   79,   14,   15,   73,   69,   91,   17},\
    {  25,   33,  139,   80,  147,  163,   66,  167,   23,   92,   70,   10},\
    { 119,   62,  145,   15,  165,   81,   28,  162,   18,   93,   71,  151},\
    {  64,   84,   28,   30,   18,   37,   62,   82,  151,   94,   13,   76},\
    {  85,   29,   63,   26,   83,   16,   36,   65,  150,  144,   95,   12},\
    {  39,   33,   86,   84,   30,   23,   64,   66,  147,   74,   10,   96},\
    {  85,   21,   31,   29,   87,   65,   38,   67,  154,  146,   11,   75},\
    {  68,   70,   31,   32,   40,   88,   24,   90,   80,  159,   11,  146},\
    {  26,   69,   71,   27,   34,   14,   89,   91,   99,   12,  144,  158},\
    {  41,   33,   32,   25,   72,   92,   90,   70,   10,  163,   78,  100},\
    {  35,   15,   93,   27,   91,   71,   28,   73,   79,  162,  145,   13},\
    {  44,   93,   37,   94,   28,   74,  145,  141,   35,   62,  102,   84},\
    {  26,   42,   75,   36,   89,   95,  144,  140,   63,   34,  103,   85},\
    {  37,   39,   30,   46,   76,   94,   74,   96,   64,  104,   86,   82},\
    {  38,   36,   75,   77,   29,   95,   97,   45,   65,    5,   83,   87},\
    {  92,   33,   76,  143,  147,   96,   39,   49,  106,   66,   41,   84},\
    { 146,   47,   97,   88,   31,   38,   77,  142,  107,   85,   40,   67},\
    {  98,   87,   78,   40,  142,  146,   31,   47,   68,   90,  108,   38},\
    { 140,   26,   99,   42,   34,   79,   83,  144,   91,   36,  109,   69},\
    {  41,   48,   40,  100,   98,   32,   80,   78,   92,    1,  110,   70},\
    {  81,   79,   34,   35,   99,   43,  101,   27,   89,  111,   71,   93},\
    {  80,   41,  143,  147,  100,   86,   49,   33,   39,   90,  112,   72},\
    {  82,   81,  101,   28,  145,  141,   35,   44,  113,   91,   37,   73},\
    {  84,   54,  104,  102,   37,   44,   46,   82,   96,  149,  141,   74},\
    {  85,   45,   36,   52,   83,  103,  105,   42,    9,  148,   97,  140},\
    {  86,   84,  104,  106,   59,   39,   49,   46,   94,  143,   76,  153},\
    {  85,   38,   45,   47,   57,  105,  107,   87,   95,  142,  152,    7},\
    {  40,   60,   48,   47,  108,  110,   90,   88,    7,  142,   78,  157},\
    { 111,   50,   43,   42,   91,   89,   34,  109,  101,  140,   79,  156},\
    {  90,   61,  110,  112,   92,   49,   48,   41,    6,  161,   80,   98},\
    { 113,   43,   44,   93,   91,  111,   35,   51,   99,  141,  160,   81},\
    {  54,  141,  113,  169,   94,   44,  118,  149,   51,  104,   82,    8},\
    {  52,   95,  109,  114,  140,   42,  168,  148,   50,   83,  105,   45},\
    {  46,  153,  149,   54,   59,   94,  128,   96,  102,   56,   84,  106},\
    {  45,  152,  148,   52,   57,   95,   97,  124,   85,   55,  107,  103},\
    { 171,   49,  153,   59,  143,  138,   96,  112,   61,   86,  104,   46},\
    { 152,   47,   57,  142,  170,  134,   97,  108,   60,   87,  105,   98},\
    {  98,   47,  157,  142,  107,  134,  170,   60,   88,   57,  110,   87},\
    { 103,   50,   99,  114,  140,   42,  168,  156,   52,  111,   89,  148},\
    { 100,  161,  157,   48,   60,   61,  136,   98,   58,   90,  108,  112},\
    { 160,   50,   43,  156,  101,   51,  116,   99,  113,   91,  109,   53},\
    { 171,  100,  161,  106,   49,  143,  138,   61,  110,   59,   92,   96},\
    { 118,  141,   44,  169,  101,   51,  102,  160,   93,   54,  111,   43},\
    { 109,  168,   50,  156,  103,  148,   52,  120,  116,   42,  124,    9},\
    { 121,   69,   63,  164,  158,   16,  150,   14,   26,  125,  117,   79},\
    { 122,  120,  156,  160,   50,   51,  111,   53,  126,   43,  114,  118},\
    { 162,   15,   14,   71,  158,   17,  123,  121,   27,  119,  115,  127},\
    { 169,  102,  113,  149,   51,  122,  160,   54,  128,   44,  116,   94},\
    {  73,  151,  123,   18,  162,   15,   62,  165,  129,   28,  117,   71},\
    { 114,   52,   50,   55,  124,  126,  116,   53,  156,  122,    9,  148},\
    {  19,   14,  125,  127,  117,  115,   16,   17,  158,   12,  123,  150},\
    {  53,   51,   54,  128,  116,  126,   56,  118,  149,  120,  160,  132},\
    {  17,   20,   18,  129,   15,  127,  117,  119,  121,  151,   13,  133},\
    {  57,  120,  152,  148,  130,   52,   55,  105,  126,  134,   45,  114},\
    {  16,  154,  150,   65,  121,   21,   19,  131,   29,  135,  115,  127},\
    { 120,   58,  130,  132,  122,   56,   55,   53,  124,  128,  116,  136},\
    {  20,   19,   22,  123,  121,  131,  133,   17,  129,  125,    3,  137},\
    { 122,   54,   56,  153,  149,   59,  132,  104,   46,  138,  118,  126},\
    { 133,  123,   20,   64,   18,  155,   23,  151,  139,  127,  119,   30},\
    {  60,   55,  124,  126,   57,   58,  136,  134,  120,  157,  152,    7},\
    {  19,   21,   24,   22,  125,  127,  137,  135,  154,  121,  133,  159},\
    {  61,   59,   58,   56,  138,  126,  136,  128,  122,  130,    6,  153},\
    {  25,   23,   22,  137,  139,  127,   20,  129,  123,  131,  163,   10},\
    { 170,   60,  157,  152,   57,  107,  108,  130,  124,  136,   47,    7},\
    {  67,   21,  154,  131,  159,   24,   68,  166,  137,   31,  125,   22},\
    {  61,  161,  157,   60,  132,  130,   58,  110,   48,  138,  134,  126},\
    { 133,  131,   70,  159,  163,   25,   24,   22,  139,  127,   32,  135},\
    { 171,   59,   61,  112,  132,  106,  161,  153,  136,   49,  128,  143},\
    {  66,  155,   72,  163,  167,  133,   25,   23,  137,  129,   33,  147},\
    { 103,   89,  109,   83,   42,   95,  168,   99,  144,  114,   34,   36},\
    { 102,   93,   44,   82,  113,  101,  145,   94,  169,   37,   35,   51},\
    {  47,   87,  107,  108,   88,   97,   98,  170,  146,   31,  134,   38},\
    {  92,   86,   49,  106,  112,   96,  100,  171,  147,  138,   59,   41},\
    {  63,   89,   83,   26,   69,   79,  140,  164,   75,  115,   34,   36},\
    {  73,   82,   62,   28,   93,  141,  165,   81,   74,   37,   44,  119},\
    {  88,   67,   31,   68,   87,  166,   77,  142,   78,   21,  135,   24},\
    {  86,   66,   33,   92,   72,  143,   76,   80,  167,   39,   49,   25},\
    { 105,  103,  114,   52,  124,   95,  168,  152,  120,   57,   55,  109},\
    { 102,  104,  118,  128,   54,   94,  169,  153,  122,   44,  113,   56},\
    {  16,   65,   63,  125,  115,   75,  121,  154,  164,   69,   26,   21},\
    { 129,  119,   62,   64,   18,  123,  165,  155,   74,   28,   23,   20},\
    { 124,  105,  107,   57,  134,   97,  130,  170,  148,   47,   52,   55},\
    { 138,   59,  104,  106,  128,  171,   96,  149,  132,   61,  112,   56},\
    { 135,   21,   67,   65,  125,   77,  131,  150,  166,   24,   68,   29},\
    {  23,  129,  139,   66,   64,   76,  167,  133,  151,   20,   30,   18},\
    { 109,  111,  116,   50,  114,  168,  160,   99,  120,   52,  103,   53},\
    {  60,  136,  134,  108,  110,  161,  170,   98,  130,   57,   61,   48},\
    {  71,   69,   14,  115,  117,   79,  164,  162,  121,   16,   15,   26},\
    { 137,   24,  135,   68,   70,  131,  166,  163,   78,   67,   22,   21},\
    { 116,  118,  113,  111,   51,  122,  156,  169,  101,   53,   54,   43},\
    {  61,  110,  112,  138,  136,  171,  132,  100,  157,   48,   49,  106},\
    {  73,  119,  117,   71,   15,  123,   81,  158,  165,   14,   28,   27},\
    {  70,   25,   72,  139,  137,  159,  167,   80,  133,   22,   24,   66},\
    {  69,  115,   63,  144,  158,  150,   26,   16,   14,  121,   79,   75},\
    {  73,  119,   62,  145,  162,  151,   28,   18,   15,  123,   74,   81},\
    { 135,   67,   68,  146,  159,  154,   24,   31,   21,   78,  131,   77},\
    {  72,  139,   66,  147,  155,  163,   23,   33,   25,  133,   80,   76},\
    { 103,  109,  114,  156,  148,  140,   42,   52,   50,   95,   99,  120},\
    { 118,  102,  113,  141,  149,  160,   51,   54,   44,  122,   94,  101},\
    { 107,  134,  108,  157,  152,  142,   47,   60,   57,   98,   97,  130},\
    { 138,  106,  112,  161,  153,  143,   61,   49,   59,   96,  100,  132}};
};

#endif /* FCC_DIFFUSION_H */

std::vector<MNP_info> *init_cluster(MNP_info &init, double r_pack, int num_mnp, XORShift<> &gen);
