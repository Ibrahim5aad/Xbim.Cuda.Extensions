#include <iostream>
#include "vector-operations.cuh"
#include "matrix-operations.cuh"
#include <cassert>
#include <cstdlib>
#include <ctime>

  
void cpu_vectorAddition(float* h_inputA, float* h_inputB, float* cpu_Output, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        cpu_Output[i] = h_inputA[i] + h_inputB[i];
    }
}


// Check result on the CPU
void cpu_matrixMultiplication(vector<long long>& a, vector<long long>& b, vector<long long>& c, int N) {
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            c[i * N + j] = tmp;
        }
    }
}


int main()
{

    // <------- Vector Addition ------->

    CudaVectorOperations cudaVectorOps;
    constexpr int N = 1000;
    clock_t time_req;

    size_t size = N * sizeof(float);

    float* h_inputA = new float[N];
    float* h_inputB = new float[N];
    float* h_output = new float[N];
    float* cpu_output = new float[N];

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_inputA[i] = static_cast<float>(i);
        h_inputB[i] = static_cast<float>(i + 100);
    }
  
    time_req = clock();
    cudaVectorOps.vectorAddition(h_inputA, h_inputB, h_output, N);
    time_req = clock() - time_req;
    std::cout << "Vector Addition On GPU " << (float)time_req / CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << std::endl;

    time_req = clock();
    cpu_vectorAddition(h_inputA, h_inputB, h_output, N);
    time_req = clock() - time_req;
    std::cout << "Vector Addition On CPU " << (float)time_req / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] h_inputA;
    delete[] h_inputB;
    delete[] h_output;


    // <------- Matrix Multiplication ------->
    CudaMatrixOperations cudaMatrixOps;

    // Host vectors
    vector<long long> h_a(N * N);
    vector<long long> h_b(N * N);
    vector<long long> h_c(N * N);
    vector<long long> h_cpu(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    time_req = clock();
    cudaMatrixOps.matrixMultiplication(h_a, h_b, h_c, N);
    time_req = clock() - time_req;
    std::cout << "Matrix Multiplication On GPU " << (float)time_req / CLOCKS_PER_SEC << " seconds" << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_c[i * N + j] << " - ";
        }
    }


    time_req = clock();
    cpu_matrixMultiplication(h_a, h_b, h_cpu, N);
    time_req = clock() - time_req;
    std::cout << "Matrix Multiplication On CPU " << (float)time_req / CLOCKS_PER_SEC << " seconds" << std::endl;
    
    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_cpu[i * N + j] << " - ";;
        }
    }*/


    return 0;
}

