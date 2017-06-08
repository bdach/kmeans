#ifndef _CPU_KMEANS_H
#define _CPU_KMEANS_H
#include "types.h"

result_t cpu_kmeans(const points_t& input, unsigned int k, float tolerance);
float calculate_distance(const points_t& input, unsigned int input_idx, const points_t& means, unsigned int means_idx);

#endif
