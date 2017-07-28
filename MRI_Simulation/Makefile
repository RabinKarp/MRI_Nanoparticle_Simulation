CXX=icpc

OBJS = rand_walk.o fcc_diffusion.o octree.o

# For performance:
CXXFLAGS = -Wall -std=c++11 -pthread -O3

# For debugging:
# CXXFLAGS = -Wall -std=c++11 -pthread -g -O0

all: octree_test T2_sim mnp_test

octree_test: octree_test.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

T2_sim: T2_sim.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

mnp_test: mnp_test.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(RNGFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f *.o octree_test T2_sim mnp_test *.tmp *~ *.out *#

.PHONY: clean all
