# Run 'module swap PrgEnv-cray PrgEnv-gnu' before compiling if c++0x is required
all:
	g++ hwinfo.cpp -I/opt/nvidia/cudatoolkit/5.0.35.102/include -L/opt/cray/nvidia/default/lib64 -lOpenCL -o hwinfo
	g++ matMult.cpp -I/opt/nvidia/cudatoolkit/5.0.35.102/include -L/opt/cray/nvidia/default/lib64 -lOpenCL -o matMult
