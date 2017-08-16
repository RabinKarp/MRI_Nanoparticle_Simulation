all: gpu_sim

gpu_sim:
	nvcc -std=c++11 --use_fast_math -lcurand -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 T2_GPU_sim.cu cuda_sim.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -o $@ $^

cpu_sim:
	g++ -std=c++11 -O2 T2_sim.cpp octree.cpp fcc_diffusion.cpp rand_walk.cpp

clean:
	rm -f *.o gpu_sim mnp_test octree_test T2_sim  *.tmp *~ *.out *#

clean_output:
	rm -f *.csv

.PHONY: clean all
