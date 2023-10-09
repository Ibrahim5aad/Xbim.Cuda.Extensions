#pragma once

#include "matrix-operations.cuh"


using namespace System;
using namespace System::Drawing;
using namespace System::Collections::Generic;

using namespace std;


namespace Xbim
{
	namespace Cuda
	{
		namespace Interop
		{

			public ref class MatrixOperations
			{

			private:
				CudaMatrixOperations* _cudaVectorOperations;
			public:
				MatrixOperations() {
					_cudaVectorOperations = new CudaMatrixOperations();
				}
				~MatrixOperations() {
					delete _cudaVectorOperations;
				}
				virtual cli::array<long long>^ MatrixOperations::MatrixMultiplication(cli::array<long long>^ mat1, cli::array<long long>^ mat2);
				virtual cli::array<float>^ MatrixOperations::MatrixMultiplication(cli::array<float>^ mat1, cli::array<float>^ mat2);
			};
		}
	}
}