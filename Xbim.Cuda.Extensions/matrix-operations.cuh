#ifndef CUDA_MATRIX_OPERATIONS_H
#define CUDA_MATRIX_OPERATIONS_H

#include <vector>

using namespace std;

class CudaMatrixOperations {
public:
    CudaMatrixOperations();
    ~CudaMatrixOperations();

    void matrixMultiplication(const vector<long long>& h_a, const vector<long long>& h_B, vector<long long>& output, int numElements);
    void matrixMultiplication(const vector<float>& h_a, const vector<float>& h_B, vector<float>& output, int numElements);
};

#endif  // CUDA_MATRIX_OPERATIONS_H
