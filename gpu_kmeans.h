#ifndef _GPU_KMEANS_H
#define _GPU_KMEANS_H
#include <vector>

#include "types.h"

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		float tolerance,
		const points_t& points,
		points_t& means,
		std::vector<unsigned int>& membership);

result_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance);

#endif
