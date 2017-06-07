#include <set>
#include <stdlib.h>
#include <time.h>

#include "common.h"

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

