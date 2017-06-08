#ifndef _TYPES_H
#define _TYPES_H

#include <vector>

typedef struct points {
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;

	points() {}
	explicit points(unsigned int size) : x(size), y(size), z(size) {}
} points_t;

typedef struct result {
	points_t means;
	std::vector<unsigned int> membership;
} result_t;

#endif
