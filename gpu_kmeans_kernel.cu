#include "cuda_runtime.h"
#include "stdio.h"

inline void on_error(cudaError_t errcode, const char *file, int line) {
	if (errcode != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(errcode), file, line);
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(ret) on_error((ret), __FILE__, __LINE__)
#define getLastCudaError() on_error(cudaGetLastError(), __FILE__, __LINE__)

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		const float *out_x,
		const float *out_y,
		const float *out_z);


extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		const float *out_x,
		const float *out_y,
		const float *out_z) {
	// no-op right now
}
