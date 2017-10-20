using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using Alea.CSharp;

namespace lms
{
    class SummatorGPU : Summator
    {        
        static DeviceMemory2D<int> memory;		
		static Gpu gpu;        
        public SummatorGPU(int channelsCount, int channelWidth, int[] detectors, int strob) 
            : base(channelsCount, channelWidth, detectors, strob)
        {
            Console.WriteLine("__SummatorGPU__");
			initGpu(detectors.Length, channelsCount);

            /*PreSaveAction = () => {
                CopySpectrumFromDevice();
            };*/
        }
		
		private static void initGpu(int ds, int chc){
			gpu = Gpu.Default;            
			memory = new DeviceMemory2D<int>(gpu.Context, new IntPtr(ds), new IntPtr(chc));
		}
				
		public override void AddValues(int[][] neutrons)
		{
			AddValuesGPU(neutrons);
		}

        public override int[][] GetSpectrum()
        {
            int[][] output = CopySpectrumFromDevice();
            /*for (int i = 0; i < detectors.Length; i++)
            {                
                deviceptr<int> ptr = memory[i].Ptr;
                gpu.For(0, channelsCount, index =>
                {
                    ptr[index] = 0;
                });
            }
            var data = Gpu.CopyToHost(memory[0]);*/
            return output;
        }

        public int[][] CopySpectrumFromDevice()
        {
            /*for (int i = 0; i < memory.Length; i++)
                spectrum[i] = Gpu.CopyToHost(memory[i]);*/
            return spectrum;
        }

        public override void ClearSpectrum()
        {
            /*for (int i = 0; i < detectors.Length; i++)
            {
                deviceptr<int> ptr = memory[i].Ptr;
                gpu.For(0, channelsCount, index =>
                {
                    ptr[index] = 0;
                });
            }*/
        }

        private static void KernelNeutronsSum(int[,] channels, int[][] frame, int maxFrameLength, int channelsCount)
        {
            int detI = blockIdx.y * blockDim.y + threadIdx.y;
            int chI = blockIdx.x * blockDim.x + threadIdx.x;

            if (detI >= frame.Length || chI >= frame[detI].Length)
                return;

            int ch = frame[detI][chI];
            channels[detI, ch] = channels[detI, ch] + 1;
        }

        private static void KernelStrobApply(int[,] spectrum, int[,] channels, int width, int height, int strob)
        {
            int detI = blockIdx.y * blockDim.y + threadIdx.y;
            int chI = blockIdx.x * blockDim.x + threadIdx.x;

            if (detI >= height || chI >= width)
                return;
            
            int ch = channels[detI, chI];            
            int k1 = ch - strob; if (k1 < 0) k1 = 0;
            int k2 = ch + strob; if (k2 > width - 1) k2 = width - 1;

            int sum = 0;
            for (int k = k1; k < k2; k++) sum += channels[detI, k];
            spectrum[detI, chI] = sum;
        }

        [GpuManaged]
        public override int[,] CalcFrameJagged(int[][] frame)
        {
            int maxEventsCount = 0;
            int maxChannel = 0;
            for (int i = 0; i < frame.Length; i++)
            {
                if (frame[i].Length > maxEventsCount)
                    maxEventsCount = frame[i].Length;
                for (int ii = 0; ii < frame[i].Length; ii++)
                    if (frame[i][ii] > maxChannel)
                        maxChannel = frame[i][ii];
            }

            int THREADS_COUNT = 1024;
            dim3 blockDim = new dim3(
                (int)Math.Ceiling((decimal)maxEventsCount / THREADS_COUNT),
                1
                );
            dim3 gridDim = new dim3(
                (int)Math.Ceiling((decimal)maxEventsCount / blockDim.x),
                frame.Length
                );
            LaunchParam lp = new LaunchParam(gridDim, blockDim);
            int[,] channels = new int[frame.Length, channelsCount];

            gpu.Launch(KernelNeutronsSum, lp, channels, frame, maxEventsCount, channelsCount);


            THREADS_COUNT = 64;
            blockDim = new dim3(
                (int)Math.Ceiling((decimal)channelsCount / THREADS_COUNT),
                1
                );
            gridDim = new dim3(
                (int)Math.Ceiling((decimal)maxEventsCount / blockDim.x),
                frame.Length
                );
            lp = new LaunchParam(gridDim, blockDim);
            int[,] spectrum = new int[frame.Length, channelsCount];

            gpu.Launch(KernelStrobApply, lp, spectrum, channels, channelsCount, frame.Length, strob);


            return spectrum;
        }

        [GpuManaged]
        public override int[][] CalcFrame2d(int[][] frame)
        {
            int s = strob;
            int c = channelsCount;

            int[][] array = new int[detectors.Length][];
            for (int i = 0; i < array.Length; i++)
                array[i] = new int[channelsCount];

            //for (int i = 0; i < frame.Length; i++)
            gpu.For(0, frame.Length, i =>
            {
                //gpu.For(0, frame[i].Length, index =>
                for (int index = 0; index < frame[i].Length; index++)
                {
                    int ch = frame[i][index];
                    int k1 = ch - s; if (k1 < 0) k1 = 0;
                    int k2 = ch + s; if (k2 > c - 1) k2 = c - 1;
                    for (int k = k1; k < k2; k++) array[i][k] += 1;
                }//);
            });
            return array;
        }

        //[GpuManaged]
        private static void AddValuesGPU(int[][] neutrons)
		{
			/*int s = strob;
			int c = channelsCount;

			for (int i = 0; i < neutrons.Length; i++)
			{
				deviceptr<int> ptr = memory[i].Ptr;
				//int[] array = new int[neutrons[i].Length];
                gpu.For(0, neutrons[i].Length, index =>
                {
                    int ch = neutrons[i][index];
                    int k1 = ch - s;
                    if (k1 < 0) k1 = 0;
                    int k2 = ch + s; if (k2 > c - 1) k2 = c - 1;
                    for (int k = k1; k < k2; k++) ptr[k] += 1;
                });
                //var data = Gpu.CopyToHost(memory[i]);
            }	*/		
		}
    }
}
