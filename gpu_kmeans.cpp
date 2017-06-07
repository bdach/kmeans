#include "common.h"
#include "gpu_kmeans.h"

#include <stdexcept>

points_t gpu_kmeans(const points_t& input, unsigned int k, float tolerance) {
	unsigned int n = input.x.size();
	if (k > n)
		throw std::invalid_argument("Number of clusters requested larger than number of points loaded");
	points_t means = initialize_means(input, k);

	// call kernel
	run_kernel(n, k, &input.x[0], &input.y[0], &input.z[0], &means.x[0], &means.y[0], &means.z[0]);
	return means;
}
