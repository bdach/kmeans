#include <exception>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "cpu_kmeans.h"
#include "gpu_kmeans.h"
#include "types.h"

#define BUF_SIZE 256

points_t parse_input(const char *filename);
void usage(char *name);
void save_output(const char *filename, const points_t& means);
void save_membership(const char *filename, const std::vector<unsigned int>& membership);

int main(int argc, char **argv) {
	char c;
	const char *input, *output;
	char *membership = nullptr;
	unsigned int k;
	bool cpu;
	float threshold = 0.01;
	unsigned int seed = time(NULL);

	while ((c = getopt(argc, argv, "i:o:k:t:gcm:s:")) != -1) {
		switch (c) {
			case 'i':
				input = optarg;
				break;
			case 'o':
				output = optarg;
				break;
			case 'k':
				k = atoi(optarg);
				break;
			case 't':
				threshold = atof(optarg);
				break;
			case 'c':
				cpu = true;
				break;
			case 'g':
				cpu = false;
				break;
			case 'm':
				membership = optarg;
				break;
			case 's':
				seed = (unsigned int)atoi(optarg);
				break;
			default:
				usage(argv[0]);
				break;
		}
	}
	if (nullptr == input || nullptr == output || k <= 1 || threshold <= 1e-8) {
		usage(argv[0]);
	}
	try {
		points_t points = parse_input(input);
		result_t result;
		if (cpu) {
			result = cpu_kmeans(points, k, threshold, seed);
		} else {
			result = gpu_kmeans(points, k, threshold, seed);
		}
		save_output(output, result.means);
		if (membership != nullptr) {
			save_membership(membership, result.membership);
		}
	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		usage(argv[0]);
	}
	return EXIT_SUCCESS;
}

void usage(char *name) {
	std::cerr << "USAGE: " << name << " -i input_file -o output_file -k cluster_count [-t threshold -gc]" << std::endl;
	std::cerr << " -i: name of a file containing 3D points in CSV format" << std::endl;
	std::cerr << " -o: name of the file to write program results to" << std::endl;
	std::cerr << " -k: number of clusters to divide points into, must be greater than 1" << std::endl;
	std::cerr << " -t: threshold for means calculation stop condition; must be greater than 1e-8" << std::endl;
	std::cerr << " -g: use GPU for computation" << std::endl;
	std::cerr << " -c: use CPU for computation (default)" << std::endl;
	std::cerr << " -m: output membership data to another file (every line contains the cluster number for each point)" << std::endl;
	std::cerr << " -s: set seed for initial means choice" << std::endl;
	exit(EXIT_FAILURE);
}

points_t parse_input(const char *filename) {
	std::ifstream input(filename);
	if (!input.good())
		throw std::invalid_argument("File not found");
	points_t points;
	char buf[BUF_SIZE], *token;
	float x, y, z;
	while (!input.eof()) {
		input.getline(buf, BUF_SIZE);
		if (buf[0] == '#') continue;
		token = strtok(buf, ",");
		if (token == nullptr) continue;
		x = atof(token);
		token = strtok(nullptr, ",");
		if (token == nullptr) continue;
		y = atof(token);
		token = strtok(nullptr, ",");
		if (token == nullptr) continue;
		z = atof(token);
		points.x.push_back(x);
		points.y.push_back(y);
		points.z.push_back(z);
	}
	return points;
}

void save_output(const char *filename, const points_t& means) {
	std::ofstream output(filename);
	if (!output.good())
		throw std::invalid_argument("Could not write means to file");
	for (unsigned int i = 0; i < means.x.size(); ++i) {
		output << means.x[i] << ",";
		output << means.y[i] << ",";
		output << means.z[i] << std::endl;
	}
}

void save_membership(const char *filename, const std::vector<unsigned int>& membership) {
	std::ofstream output(filename);
	if (!output.good())
		throw std::invalid_argument("Could not write membership data to file");
	for (auto m : membership) {
		output << m << std::endl;
	}
}
