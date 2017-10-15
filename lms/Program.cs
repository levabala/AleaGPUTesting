using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using System.Diagnostics;

namespace lms
{
    class Program
    {
        static int length = 10000000;
        static int rounds = 100;
        static int chunks = 10;

        static void Main(string[] args)
        {
            Random rnd = new Random();
            double[] randomArr = new double[length];
            for (int i = 0; i < length; i++)
                randomArr[i] = Math.Round(rnd.NextDouble() * 10, 3);

            Console.WriteLine("Elements count: {0}\nRounds count: {1}\n", length, rounds);

            Stopwatch sw = new Stopwatch();

            sw.Start();
            DeviceMemory<double> memory = alea(randomArr);
            sw.Stop();
            long gpuAllocating = sw.ElapsedMilliseconds;
            Console.WriteLine("Device memory allocated: {0}", gpuAllocating);
            sw.Reset();

            sw.Start();
            for (int i = 0; i < rounds; i++)
                alea2(memory);            
            sw.Stop();
            long gpuCalc = sw.ElapsedMilliseconds;

            var data = Gpu.CopyToHost(memory);
            
            sw.Start();
            double[] array = new double[length];
            for (int i = 0; i < array.Length; i++)
                array[i] = randomArr[i];
            sw.Stop();
            long cpuAllocating = sw.ElapsedMilliseconds;
            Console.WriteLine("Host memory allocated: {0}\n", cpuAllocating);
            sw.Reset();

            sw.Start();
            for (int ii = 0; ii < rounds; ii++)
                for (int i = 0; i < array.Length; i++)
                    array[i] *= (Math.Sin(array[i]));
            sw.Stop();
            long cpuCalc = sw.ElapsedMilliseconds;

            double[] differenceChunks = new double[chunks];
            double totalDiff = 0;
            Parallel.For(0, length, i =>
            {
                double diff = Math.Abs(array[i] - data[i]);
                int chunkIndex = i / (length / chunks);
                differenceChunks[chunkIndex] += diff;
                totalDiff += diff;
            });

            Console.WriteLine("Gpu elapsed: {0}\nCpu elapsed: {1}\n", gpuCalc, cpuCalc);
            Console.WriteLine("Difference: {0}", totalDiff);
            Console.WriteLine("Average diff: {0}", totalDiff / length);
            Console.WriteLine("Average diff per round: {0}", totalDiff / length / rounds);
            for (int i = 0; i < chunks; i++)
                Console.WriteLine("DifferenceChunk{0}: {1}", i, differenceChunks[i]);
            Console.ReadKey();
        }

        static DeviceMemory<double> alea(double[] array)
        {
            DeviceMemory<double> memory = Gpu.Default.AllocateDevice<double>(length);
            deviceptr<double> pointer = memory.Ptr;            
            Gpu.Default.For(0, length, i => pointer[i] = array[i]);

            //var data = Gpu.CopyToHost(memory);

            return memory;
        }

        static void alea2(DeviceMemory<double> memory)
        {
            deviceptr<double> pointer = memory.Ptr;
            Gpu.Default.For(0, length, i => pointer[i] *= (DeviceFunction.Sin(pointer[i])));

            //var data = Gpu.CopyToHost(memory);
        }
    }
}
