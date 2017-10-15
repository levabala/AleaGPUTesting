using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    class SummatorGPU : Summator
    {
        public SummatorGPU(int channelsCount, int channelWidth, List<int> detectors, int strob) 
            : base(channelsCount, channelWidth, detectors, strob)
        {

        }
    }
}
