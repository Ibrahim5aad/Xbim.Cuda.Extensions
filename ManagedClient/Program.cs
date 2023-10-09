using System.Diagnostics;
using System.Numerics;
using Xbim.Cuda.Interop;

namespace ManagedClient
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //var vectorOps = new VectorOperations();

            //var a = GetRandomList();
            //var b = GetRandomList();

            //Stopwatch sw = new Stopwatch();
            //sw.Start();
            //var c = vectorOps.VectorAddition(a, b);
            //sw.Stop();
            //Console.WriteLine("Managed Call to VectorAddition took {0}ms", sw.ElapsedMilliseconds);


            var matrixOps = new MatrixOperations();
            Console.Write("> Enter Matrix Dimension: ");
            var dimInput = Console.ReadLine();
            var dim = int.Parse(dimInput!);
            Stopwatch sw2 = new Stopwatch();
            sw2.Start();
            long[] mat1 = GetRandomMatrix(dim);
            long[] mat2 = GetRandomMatrix(dim);
            var mat = matrixOps.MatrixMultiplication(mat1, mat2);
            sw2.Stop();
            Console.WriteLine(" ------------------------------------------- ");

            Console.WriteLine($"({dim}x{dim}) * ({dim}x{dim}) Matrix Multiplication");
            var gpuTime = sw2.ElapsedMilliseconds;
            Console.WriteLine(" ------------------------------------------- ");
            Console.WriteLine(" -- GPU: took {0}ms", gpuTime);

            sw2.Restart();
            CPU_MatrixMultiplicationAssertion(mat1, mat2, mat, dim);
            sw2.Stop();

            var cpuTime = sw2.ElapsedMilliseconds;
            Console.WriteLine(" -- CPU: took {0}ms", cpuTime);
            Console.WriteLine(" ------------------- ");
            var gains = cpuTime / gpuTime;
            Console.WriteLine("Performance improvement: {0}x", gains);
            Console.ReadKey();
        }


        static void CPU_MatrixMultiplicationAssertion(long[] a, long[] b, long[] c, long N)
        {
            // For every row...
            for (int i = 0; i < N; i++)
            {
                // For every column...
                for (int j = 0; j < N; j++)
                {
                    // For every element in the row-column pair
                    long tmp = 0;
                    for (int k = 0; k < N; k++)
                    {
                        // Accumulate the partial results
                        tmp += a[i * N + k] * b[k * N + j];
                    }

                    // Check against the CPU result
                    if (tmp != c[i * N + j])
                    {
                        throw new Exception("Verification failed at element (" + i + ", " + j + ")!");
                    }
                }
            }
        }



        private static List<float> GetRandomList()
        {

            var rand = new Random();
            var rtnlist = new List<float>();

            for (int i = 0; i < 1000000; i++)
            {
                rtnlist.Add(rand.Next(1000));
            }
            return rtnlist;
        }

        private static long[] GetRandomMatrix(int dimension)
        {
            var rand = new Random();
            var rtnlist = new long[dimension * dimension];

            for (int i = 0; i < dimension * dimension; i++)
            {
                rtnlist[i] = rand.Next(1000);
            }
            return rtnlist;

        }
    }
}