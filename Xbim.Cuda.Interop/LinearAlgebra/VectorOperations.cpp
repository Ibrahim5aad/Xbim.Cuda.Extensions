
#include "VectorOperations.h" 
 
using namespace System;
using namespace System::Linq;

namespace Xbim
{
	namespace Cuda
	{
		namespace Interop
		{
			IEnumerable<float>^ VectorOperations::VectorAddition(IEnumerable<float>^ vector1, IEnumerable<float>^ vector2)
			{
				int n = System::Math::Min(Enumerable::Count(vector1), Enumerable::Count(vector2));

				float* v1 = new float[n];
				float* v2 = new float[n];
				float* v3 = new float[n];
				
				for (size_t i = 0; i < n; i++)
				{
					v1[i] = Enumerable::ElementAt(vector1, i);
					v2[i] = Enumerable::ElementAt(vector2, i);
				}
				 
				_cudaVectorOperations->vectorAddition(v1, v2, v3, n);

				List<float>^ result = gcnew List<float>(n);

				for (size_t i = 0; i < n; i++)
				{
					result->Add(v3[i]);
				}

				return result;
			};
		}
	}
}