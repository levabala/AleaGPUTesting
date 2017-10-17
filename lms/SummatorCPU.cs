using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    class SummatorCPU : Summator
    {
		/*public static double[][] channels; //detector - channel_value
		public static int channelsCount, channelWidth, strob;
		public static int[] detectors;*/
        public SummatorCPU(int channelsCount, int channelWidth, int[] detectors, int strob) 
            : base(channelsCount, channelWidth, detectors, strob)
        {
            
        }

        public override int[][] GetSpectrum()
        {
            int[][] output = spectrum.ToArray();
            spectrum = createSpectrumArray();
            return output;
        }

        public override int[][] CalcFrame(int[][] frame)
        {
            int[][] spectr = createSpectrumArray();
            for (int detector = 0; detector < frame.Length; detector++)
            {
                int[] s = frame[detector];
                //for (int i = 0; i < s.Length;)
                Parallel.For(0, s.Length, i =>
                {
                    int ch = s[i];
                    int k1 = ch - strob; if (k1 < 0) k1 = 0;
                    int k2 = ch + strob; if (k2 > channelsCount - 1) k2 = channelsCount - 1;
                    for (int k = k1; k < k2; k++) spectr[detector][k] += 1;
                });
            }
            return spectr;
        }

        public override void AddValues(int[][] neutrons)
        {            
            for (int detector = 0; detector < neutrons.Length; detector++)            
            {
                int[] s = neutrons[detector];
                Parallel.For(0, s.Length, i =>
                {
                    int ch = s[i];
                    int k1 = ch - strob; if (k1 < 0) k1 = 0;
                    int k2 = ch + strob; if (k2 > channelsCount - 1) k2 = channelsCount - 1;
                    for (int k = k1; k < k2; k++) spectrum[detector][k] += 1;
                });
            }
        }
    }
}
