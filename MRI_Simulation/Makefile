CXX=icpc

OBJS = rand_walk.o fcc_diffusion.o octree.o

# For performance:
CXXFLAGS = -Wall -std=c++11 -pthread -O3

# For debugging:
# CXXFLAGS = -Wall -std=c++11 -pthread -g -O0

all: p_sweep_sim

p_sweep_sim: p_sweep_sim.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f *.o octree_test p_sweep_sim *.tmp *~ *.out *#

.PHONY: clean all
