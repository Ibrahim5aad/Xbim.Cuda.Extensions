#ifndef CUDA_VECTOR_OPERATIONS_H
#define CUDA_VECTOR_OPERATIONS_H

#include <algorithm>
#include <vector>

using namespace std;

class CudaVectorOperations {
public:
    CudaVectorOperations();
    ~CudaVectorOperations();

    void scalarMultiplication(float* input, float scalar, float* output, int numElements);
    void vectorAddition(float* inputA, float* inputB, float* output, int numElements);
};

#endif  // CUDA_VECTOR_OPERATIONS_H
