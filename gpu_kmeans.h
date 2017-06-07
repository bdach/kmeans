#ifndef _GPU_KMEANS_H
#define _GPU_KMEANS_H
#include "types.h"

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		const float *out_x,
		const float *out_y,
		const float *out_z);

points_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance);

#endif
