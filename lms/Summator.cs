using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    abstract class Summator
    {
        public double[][] channels; //detector - channel_value
        public int channelsCount, channelWidth, strob;
        public int[] detectors;
        public Summator(int channelsCount, int channelWidth, int[] detectors, int strob)
        {
            this.channelsCount = channelsCount;
            this.channelWidth = channelWidth;
            this.detectors = detectors;
            this.strob = strob;
        }

        virtual public void AddValues(int[][] neutrons) //detector - neutron_channel
        {

        }                
    }
}
