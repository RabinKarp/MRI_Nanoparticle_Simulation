# Compiling local simulation files

L_OBJS=T2_GPU_sim.o gpu_random.o cuda_sim.o fcc_diffusion.o rand_walk.o octree.o 
PARAMS=parameters.h
#==============================================================================
# Compiling CUDA utilities from git submodule

UTIL_FILE=utilities/.dependencies
UTIL_LIST=$(shell cat ${UTIL_FILE})
U_OBJS=$(addprefix utilities/,$(UTIL_LIST))

#==============================================================================
OBJS=$(L_OBJS) $(U_OBJS) 
CXX=nvcc

cuda_flags= -std=c++11 \
	   -gencode arch=compute_35,code=compute_35 \
       	   -gencode arch=compute_35,code=sm_35 \
   	   -ccbin=icpc -lcurand -lcublas -lcusolver -restrict\
           -Xcompiler "-pthread -O3 -mkl"

all: module_loader

module_loader:
	sh build.sh	

gpu_sim: $(OBJS) $(PARAMS)
	$(CXX) $(cuda_flags) -o $@.out $(OBJS) 

%.o: %.cpp $(PARAMS)
	$(CXX) -c $(cuda_flags) $< -o $@

%.o: %.cu $(PARAMS)
	$(CXX) -c $(cuda_flags) $< -o $@

# Printing statistics about the simulation box
sim_box_test: $(PARAMS)
	g++ -std=c++11 octree.cpp SimulationBox.cpp BacteriaBox.cpp SimboxTest.cpp -o simstat.out

clean:
	rm -f *.o *.tmp *~ *.out *#

clean_output:
	rm -f *.csv

.PHONY: clean all
