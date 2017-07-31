CXX=nvcc

# For performance:
CXXFLAGS = -std=c++11 -lcurand

all: gpu_sim

gpu_sim:
	nvcc -std=c++11 -lcurand cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -o $@ $^

clean:
	rm -f *.o gpu_sim  *.tmp *~ *.out *#

.PHONY: clean all
