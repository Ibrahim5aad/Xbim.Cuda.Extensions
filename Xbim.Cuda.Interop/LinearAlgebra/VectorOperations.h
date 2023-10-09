#pragma once

#include "vector-operations.cuh"

using namespace System;
using namespace System::Collections::Generic;


namespace Xbim
{
	namespace Cuda
	{
		namespace Interop
		{

			public ref class VectorOperations
			{

			private:
				CudaVectorOperations* _cudaVectorOperations;
			public:
				VectorOperations() {
					_cudaVectorOperations = new CudaVectorOperations();
				}
				~VectorOperations() {
					delete _cudaVectorOperations;
				}
				virtual IEnumerable<float>^ VectorAddition(IEnumerable<float>^ vector1, IEnumerable<float>^ vector2);
			};
		}
	}
}