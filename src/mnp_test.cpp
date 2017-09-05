/**
 * This simple program generates the debug file for a variety of MNP
 * configurations - intracellular, extracellular, both, along with
 * clustered and unclustered MNPs.
 */

 #include <thread>
 #include <iostream>
 #include <mutex>
 #include <cassert>
 #include "octree.h"
 #include "fcc_diffusion.h"
 #include "parameters.h"

using namespace std;

int main() {

#ifdef CLUSTERED
  cout << "Set flag CLUSTERED" << endl;
#else
  cout << "Set flag UNCLUSTERED" << endl;
#endif

#if defined EXTRACELLULAR
  cout << "Set flag EXTRACELLULAR" << endl;
#elif defined INTRACELLULAR
  cout << "Set flag INTRACELLULAR" << endl;
#else
  cout << "Set flag INTRA_EXTRA" << endl;
#endif

  std::random_device rd;
  XORShift<> gen(time(NULL) + rd());
  FCC *lattice = new FCC(D_cell, D_extra, P_expr);
  std::vector<MNP_info> *mnps = lattice->init_mnps(gen);
}
