CXX=icpc

OBJS = rand_walk.o fcc_diffusion.o octree.o

# For performance:
CXXFLAGS = -Wall -std=c++11 -pthread -O3

# For debugging:
# CXXFLAGS = -Wall -std=c++11 -pthread -g -O0

# Might help PRNG stuff???
RNGFLAGS = -fno-move-loop-invariants -fno-unroll-loops
RNGFLAGS =

all: octree_test diffusion_test T2_sim

octree_test: octree_test.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

diffusion_test: diffusion_test.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

T2_sim: T2_sim.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f *.o octree_test diffusion_test T2_sim *.tmp *~ *.out

.PHONY: clean all
