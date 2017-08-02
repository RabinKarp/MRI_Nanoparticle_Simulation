#include <iostream>
#include "rand_walk.h"
#include "fcc_diffusion.h"
#include "parameters.h"

using namespace std;

int main(int argc, char** argv) {
    cout << "Initializing lattice" << endl;
    FCC lattice(D_cell, D_extra, P_expr);

    std::random_device rd;
    XORShift<> gen(time(NULL) + rd());
    for(int i = 0; i < 100; i++) {
        water_info w;
        w.x = gen.rand_pos_double() * bound;
        w.y = gen.rand_pos_double() * bound;
        w.z = gen.rand_pos_double() * bound;

        lattice.update_nearest_cell(&w);
        int init = w.nearest;
        lattice.update_nearest_cell_full(&w);
        int fin = w.nearest;

        cout << init << " | " << fin << endl;

        if(init != fin) {
            cout << "Error: " << w.x << " " << w.y << " " << w.z
                << " | " << init << " | " << fin << endl;
        }
    }
}
