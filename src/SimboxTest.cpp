#include "BacteriaBox.h"
#include "parameters.h"
#include "xorshift.h"
#include <ctime>
#include <cstdlib>
#include <random>

using namespace std;

int main(int argc, char** argv) {
    std::random_device rd; 
    XORShift<uint64_t> gen(time(NULL) + rd());
    BacteriaBox simBox(num_cells, num_water, &gen);

    simBox.populateSimulation();
    simBox.print_simulation_stats();
}
