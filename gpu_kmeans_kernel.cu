#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <vector>

inline void on_error(cudaError_t errcode, const char *file, int line) {
	if (errcode != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(errcode), file, line);
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(ret) on_error((ret), __FILE__, __LINE__)
#define getLastCudaError() on_error(cudaGetLastError(), __FILE__, __LINE__)
#define NUM_THREADS 256

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		float tolerance,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		float *out_x,
		float *out_y,
		float *out_z);

__global__ void calculate_distances(unsigned int n,
		unsigned int k,
		float **points,
		float **means,
		unsigned int *membership,
		unsigned char *subdelta)
{
	extern __shared__ unsigned char membership_changed[];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if (idx >= n) return;
	float min_dist = INFINITY;
	unsigned int cluster = 0;
	for (unsigned int i = 0; i < k; ++i) {
		float dx = points[0][idx] - means[0][i];
		float dy = points[1][idx] - means[1][i];
		float dz = points[2][idx] - means[2][i];
		float dist = dx * dx + dy * dy + dz * dz;
		if (dist < min_dist) {
			min_dist = dist;
			cluster = i;
		}
		__syncthreads(); // end of uncertain branch
	}
	membership_changed[tid] = membership[idx] == cluster;
	membership[idx] = cluster;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			membership_changed[tid] += membership_changed[tid + s];
		__syncthreads();
	}
	if (tid == 0) {
		subdelta[blockIdx.x] = membership_changed[0];
	}
}

extern "C" void run_kernel(unsigned int n,
		unsigned int k,
		float tolerance,
		const float *in_x,
		const float *in_y,
		const float *in_z,
		float *out_x,
		float *out_y,
		float *out_z)
{
	const unsigned int points_size = n * sizeof(float);
	const float *points[3] = {in_x, in_y, in_z};
	float *points_d[3];
	float **d_points_d;
	for (int i = 0; i < 3; ++i) {
		checkCudaErrors(cudaMalloc((void **)&points_d[i], points_size));
		checkCudaErrors(cudaMemcpy(points_d[i], points[i], points_size, cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void **)&d_points_d, sizeof(points_d)));
	checkCudaErrors(cudaMemcpy(d_points_d, points_d, sizeof(points_d), cudaMemcpyHostToDevice));

	const unsigned int means_size = k * sizeof(float);
	float *means[3] = {out_x, out_y, out_z};
	float *means_d[3];
	float **d_means_d;
	for (unsigned int i = 0; i < 3; ++i) {
		checkCudaErrors(cudaMalloc((void **)&means_d[i], means_size));
		checkCudaErrors(cudaMemcpy(means_d[i], means[i], means_size, cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void **)&d_means_d, sizeof(means_d)));
	checkCudaErrors(cudaMemcpy(d_means_d, means_d, sizeof(means_d), cudaMemcpyHostToDevice));

	unsigned int block_count = ceil((float)n / NUM_THREADS);
	unsigned int subdelta_size = block_count * sizeof(unsigned char);
	unsigned int shared_subdelta_size = NUM_THREADS * sizeof(unsigned char);
	unsigned char *d_subdelta; // WARNING: This works because NUM_THREADS is 256
	checkCudaErrors(cudaMalloc((void **)&d_subdelta, subdelta_size));

	const unsigned int membership_size = n * sizeof(unsigned int);
	unsigned int *new_membership, *d_new_membership;
	new_membership = (unsigned int *)malloc(membership_size);
	checkCudaErrors(cudaMalloc((void **)&d_new_membership, membership_size));

	unsigned int delta = n;
	while (((float)delta / n) > tolerance) {
		delta = 0;
		calculate_distances<<< block_count, NUM_THREADS, shared_subdelta_size >>>(n, k, d_points_d, d_means_d, d_new_membership, d_subdelta);
		getLastCudaError();
		thrust::device_ptr<unsigned char> ptr(d_subdelta);
		delta = thrust::reduce(ptr, ptr + block_count);
		std::vector<unsigned int> counts(k);
		for (unsigned int j = 0; j < k; ++j) {
			means[0][j] = 0;
			means[1][j] = 0;
			means[2][j] = 0;
		}
		checkCudaErrors(cudaMemcpy(new_membership, d_new_membership, membership_size, cudaMemcpyDeviceToHost));
		for (unsigned int i = 0; i < n; ++i) {
			unsigned int cluster = new_membership[i];
			means[0][cluster] += points[0][i];
			means[1][cluster] += points[1][i];
			means[2][cluster] += points[2][i];
			counts[cluster] += 1;
		}
		for (unsigned int j = 0; j < k; ++j) {
			means[0][j] /= counts[j];
			means[1][j] /= counts[j];
			means[2][j] /= counts[j];
		}
		for (unsigned int i = 0; i < 3; ++i) {
			checkCudaErrors(cudaMemcpy(means_d[i], means[i], means_size, cudaMemcpyHostToDevice));
		}
	}

	for (unsigned int i = 0; i < 3; ++i) {
		checkCudaErrors(cudaFree(points_d[i]));
		checkCudaErrors(cudaFree(means_d[i]));
	}
	checkCudaErrors(cudaFree(d_points_d));
	checkCudaErrors(cudaFree(d_means_d));
	checkCudaErrors(cudaFree(d_new_membership));
	checkCudaErrors(cudaFree(d_subdelta));
	free(new_membership);
}
