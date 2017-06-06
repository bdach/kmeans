#include <set>
#include <stdlib.h>
#include <time.h>

#include "types.h"

points_t initialize_means(const points_t& input, unsigned int k);

points_t cpu_kmeans(const points_t& input, int k, float tolerance) {
	// TODO: check that k is less than the number of points
	points_t means = initialize_means(input, k);
	return means;
}

points_t initialize_means(const points_t& input, unsigned int k) {
	srand(time(NULL));
	points_t means;
	std::set<int> indices;
	int count = input.x.size();
	while (indices.size() < k) {
		int idx = rand() % count;
		indices.insert(idx);
	}
	for (auto index : indices) {
		means.x.push_back(input.x[index]);
		means.y.push_back(input.y[index]);
		means.z.push_back(input.z[index]);
	}
	return means;
}

