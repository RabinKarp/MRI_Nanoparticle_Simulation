all: gpu_sim

gpu_sim:
	nvcc -std=c++11 --use_fast_math -lcurand -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 cuda_test.cu fcc_diffusion.cpp rand_walk.cpp octree.cpp -o $@ $^

clean:
	rm -f *.o gpu_sim mnp_test octree_test T2_sim  *.tmp *~ *.out *#

clean_output:
	rm -f *.csv

.PHONY: clean all
