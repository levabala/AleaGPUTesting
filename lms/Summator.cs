using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    abstract class Summator
    {		
		public static double[][] channels; //detector - channel_value
		public static int channelsCount, channelWidth, strob;
		public static int[] detectors;
        public Summator(int chcount, int chwidth, int[] dets, int strob_value)
        {
			channelsCount = chcount;
			channelWidth = chwidth;
			detectors = dets;
			strob = strob_value;
			channels = new double[dets.Length][];
			for (int i = 0; i < channels.Length; i++)
				channels[i] = new double[channelsCount];
        }

		virtual public void AddValues(int[][] neutrons) //detector - neutron_channel
        {

        }                
    }
}
