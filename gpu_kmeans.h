#ifndef _GPU_KMEANS_H
#define _GPU_KMEANS_H
#include "types.h"

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		float tolerance,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		float *out_x,
		float *out_y,
		float *out_z);

points_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance);

#endif
