using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;

namespace lms
{
    class SummatorGPU : Summator
    {		
		static DeviceMemory<double>[] memory;		
		static Gpu gpu;
        public SummatorGPU(int channelsCount, int channelWidth, int[] detectors, int strob) 
            : base(channelsCount, channelWidth, detectors, strob)
        {
			initGpu(detectors.Length, channelsCount);
        }
		
		private static void initGpu(int ds, int chc){
			gpu = Gpu.Default;
			memory = new DeviceMemory<double>[ds];
			for (int i = 0; i < ds; i++)
			{
				memory[i] = gpu.AllocateDevice<double>(chc);
				deviceptr<double> ptr = memory[i].Ptr;
			}						
		}
				
		public override void AddValues(int[][] neutrons)
		{
			AddValuesGPU(neutrons);
		}
		
		private static void AddValuesGPU(int[][] neutrons)
		{
			int s = strob;
			int c = channelsCount;

			for (int i = 0; i < neutrons.Length; i++)
			{
				deviceptr<double> ptr = memory[i].Ptr;
				int[] array = new int[neutrons[i].Length];
				gpu.For(0, array.Length, index =>
				{
					int ch = array[index];
					int k1 = ch - s;
					if (k1 < 0) k1 = 0;
					int k2 = ch + s; if (k2 > c - 1) k2 = c - 1;
					for (int k = k1; k < k2; k++) ptr[k] += 1;
				});				
			}			
		}
    }
}
