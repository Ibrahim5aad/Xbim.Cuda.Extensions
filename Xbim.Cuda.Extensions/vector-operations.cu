#include "vector-operations.cuh"
#include <cuda_runtime.h>

#include <functional>
#include <iostream>


__global__ void scalarMultiplicationKernel(float* input, float scalar, float* output, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = scalar * input[i];
    }
}

__global__ void vectorAdditionKernel(float* inputA, float* inputB, float* output, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = inputA[i] + inputB[i];
    }
}

CudaVectorOperations::CudaVectorOperations() {
}

CudaVectorOperations::~CudaVectorOperations() {
    
}


void CudaVectorOperations::scalarMultiplication(float* input, float scalar, float* output, int numElements) {
    size_t size = numElements * sizeof(float);

    float* d_input, * d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input array to GPU
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Launch scalar multiplication kernel
    scalarMultiplicationKernel<<<gridSize, blockSize>>> (d_input, scalar, d_output, numElements);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void CudaVectorOperations::vectorAddition(float* inputA, float* inputB, float* output, int numElements) {
    size_t size = numElements * sizeof(float);

    float* d_input, * d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Allocate memory on GPU if not allocated already
    if (d_input == nullptr) {
        cudaMalloc((void**)&d_input, size);
    }
    if (d_output == nullptr) {
        cudaMalloc((void**)&d_output, size);
    }

    // Copy input arrays to GPU
    cudaMemcpy(d_input, inputA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, inputB, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Launch vector addition kernel
    vectorAdditionKernel<<<gridSize, blockSize>>>(d_input, d_output, d_output, numElements);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
