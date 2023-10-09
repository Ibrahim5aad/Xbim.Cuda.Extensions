#include "MatrixOperations.h" 
#include <vector>
#include <algorithm>
#include "../Utils/ArrayMarshal.h"
#include <cmath>
#include <iostream>


using namespace std;
using namespace System;

namespace Xbim
{
	namespace Cuda
	{
		namespace Interop
		{
			cli::array<long long>^ MatrixOperations::MatrixMultiplication(cli::array<long long>^ mat1, cli::array<long long>^ mat2) {
				 
				vector<long long> vec1 = pincpy_v_a<long long>(mat1);
				vector<long long> vec2 = pincpy_v_a<long long>(mat2);
				vector<long long> output(vec1.size());
				 
				_cudaVectorOperations->matrixMultiplication(vec1, vec2, output, sqrt(vec1.size()));

				return pincpy_a_v(output);
			}
		
			cli::array<float>^ MatrixOperations::MatrixMultiplication(cli::array<float>^ mat1, cli::array<float>^ mat2) {

				vector<float> vec1 = pincpy_v_a<float>(mat1);
				vector<float> vec2 = pincpy_v_a<float>(mat2);
				vector<float> output(vec1.size());

				_cudaVectorOperations->matrixMultiplication(vec1, vec2, output, sqrt(vec1.size()));

				return pincpy_a_v(output);
			}
		}
	}
}