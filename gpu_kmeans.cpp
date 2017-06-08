#include "common.h"
#include "gpu_kmeans.h"

#include <stdexcept>

points_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance) {
	unsigned int n = input.x.size();
	if (k > n)
		throw std::invalid_argument("Number of clusters requested larger than number of points loaded");
	points_t means = initialize_means(input, k);

	// call kernel
	run_kernel(n, k, tolerance, input, means);
	return means;
}
