#ifndef _GPU_KMEANS_H
#define _GPU_KMEANS_H
#include "types.h"

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		float tolerance,
		const points_t& points,
		points_t& means);

points_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance);

#endif
