#include <limits>
#include <set>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>

#include "types.h"

points_t initialize_means(const points_t& input, unsigned int k);
float calculate_distance(const points_t& input, unsigned int input_idx, const points_t& means, unsigned int means_idx);

points_t cpu_kmeans(const points_t& input, unsigned int k, float tolerance) {
	unsigned int n = input.x.size();
	if (k > n)
		throw std::invalid_argument("Number of clusters requested larger than number of points loaded");
	points_t means = initialize_means(input, k);

	unsigned int delta = n;
	std::vector<unsigned int> membership(n);

	while (((double)delta / n) > tolerance) {
		points_t new_means(k);
		std::vector<unsigned int> means_count(k);
		delta = 0;
		for (unsigned int i = 0; i < n; ++i) {
			float min_dist = std::numeric_limits<float>::infinity();
			unsigned int idx = 0;
			for (unsigned int j = 0; j < k; ++j) {
				float distance = calculate_distance(input, i, means, j);
				if (distance < min_dist) {
					min_dist = distance;
					idx = j;
				}
			}
			if (membership[i] != idx) {
				delta++;
				membership[i] = idx;
			}
			new_means.x[idx] += input.x[i];
			new_means.y[idx] += input.y[i];
			new_means.z[idx] += input.z[i];
			means_count[idx]++;
		}
		for (unsigned int j = 0; j < k; ++j) {
			new_means.x[j] /= means_count[j];
			new_means.y[j] /= means_count[j];
			new_means.z[j] /= means_count[j];
		}
		means = new_means;
	}
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

float calculate_distance(const points_t& input, unsigned int input_idx, const points_t& means, unsigned int means_idx) {
	float dx = input.x[input_idx] - means.x[means_idx];
	float dy = input.y[input_idx] - means.y[means_idx];
	float dz = input.z[input_idx] - means.z[means_idx];
	return dx * dx + dy * dy + dz * dz;
}
