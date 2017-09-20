#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>

#include "cuda.h"

#include "parameters.h"
#include "cuda_sim.h"

using namespace std;

/*
 * Generates a detailed filename for an output file, containing information
 * about the parameters used to generate it.
 */
std::string generate_base_filename()
{
    std::string filename("tau=");
    filename += std::to_string((unsigned)(tau * 1e9));
    filename += "ps_T-e=";
    filename += std::to_string((unsigned)(2*taucp));
    filename += "largedipole";

#ifdef EXPLICIT
    filename += "_ex_";
#endif

#ifndef FULL_BOUNDARIES
    filename += "_border=";
    filename += std::to_string((unsigned)(border));
    filename += "um_";
#endif

    filename += std::to_string((unsigned)(time(NULL)));
    filename += ".csv";
    return filename;
}

int main(int argc, char** argv) {
    std::string fName = generate_base_filename();
    simulateWaters(fName);
}
