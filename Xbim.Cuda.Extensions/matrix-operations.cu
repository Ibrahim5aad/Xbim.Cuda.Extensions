#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include "matrix-operations.cuh"


__global__ void matrixMul(const long long* a, const long long* b, long long* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        c[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}

__global__ void matrixMul(const float* a, const float* b, float* c, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        c[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}



CudaMatrixOperations::CudaMatrixOperations() {
}

CudaMatrixOperations::~CudaMatrixOperations() {

}

void CudaMatrixOperations::matrixMultiplication(const vector<long long>& h_a, const vector<long long>& h_b, vector<long long>& output, int numElements) {

    size_t bytes = numElements * numElements * sizeof(long long);

    // Allocate device memory
    long long* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = (numElements / THREADS) + 1;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMul<<<blocks, threads>>> (d_a, d_b, d_c, numElements);

    // Copy back to the host
    cudaMemcpy(output.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void CudaMatrixOperations::matrixMultiplication(const vector<float>& h_a, const vector<float>& h_b, vector<float>& output, int numElements) {

    size_t bytes = numElements * numElements * sizeof(float);

    // Allocate device memory
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = (numElements / THREADS) + 1;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMul<<<blocks, threads>>> (d_a, d_b, d_c, numElements);

    // Copy back to the host
    cudaMemcpy(output.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}