using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace lms
{
    class Parser
    {
        public static void Parse(
            List<string> filesNames, int strob, int channelsCount, int channelWidth, 
            int framesCount, int tau, double[] kt, int[] detectors, int channel0,
			ParameterizedThreadStart SummatorCall)
		{
			int frame = 0;
			int save_count = 0;
			HashSet<int> detectorsHashSet = new HashSet<int>(detectors);			

			for (int k = 0; k < filesNames.Count; k++)
			{
				string nam = filesNames[k];

				FileStream fs = File.OpenRead(nam);
				long len = fs.Length;

				BinaryReader br = new BinaryReader(fs);
				long pos = 0;

				Stopwatch sw = Stopwatch.StartNew();

				float speed_mbs = 0;
				float speed_x = 0;
				int lastFrameNeutrons = 0;

				int[] kk = new int[0x100];

				long fbeg = 0;
				long fend = 0;
				long f0 = 0;
				long f1 = 0;
				long f2 = 0;
				long f3 = 0;
				long f4 = 0;
				long f5 = 0;
				long f6 = 0;
				long f7 = 0;
				long f8 = 0;				

				List<int>[] neutrons = new List<int>[detectors.Length];
				for (int d = 0; d < neutrons.Length; d++)
				{
					neutrons[d] = new List<int>();
				}

				int neutronsCount = 0;

				while (pos < len)
				{
					Stopwatch sw2 = Stopwatch.StartNew();

					byte[] buf = br.ReadBytes(1000000);
					pos += buf.Length;					

					for (int i = 0; i < buf.Length; i++)
					{
						int lo = (buf[i++]) | (buf[i++] << 8) | (buf[i++] << 16);

						byte hi = buf[i];
						kk[hi]++;												

						switch (hi)
						{
							case 0xf0: f0 = lo; break;
							case 0xf1: f1 = lo; break;
							case 0xf2: f2 = (long)lo | ((long)f4) << 24; break;
							case 0xf3: f3 = (long)lo | ((long)f4) << 24; break;
							case 0xf4: f4++; break;
							case 0xf5: f5 = (long)lo | ((long)f4) << 24; break;
							case 0xf6: f6 = (long)lo | ((long)f4) << 24; break;
							case 0xf7: f7 = (long)lo | ((long)f4) << 24; break;
							case 0xf8: f8 = (long)lo | ((long)f4) << 24; break;
							case 0xfb: fbeg = (f0 | ((long)(f4) << 24)); break;
							case 0xfa:
								fend = (long)lo | ((long)(f4) << 24); frame += 1;
								if (frame % framesCount == 0)
								{
									float spec_time = (float)(fend * 16e-9);									

									int neutronsDelta = neutronsCount - lastFrameNeutrons;
									lastFrameNeutrons = neutronsCount;

									int[][] array = new int[neutrons.Length][];
									for (int ii = 0; ii < array.Length; ii++)
										array[ii] = neutrons[ii].ToArray();

									Thread summatorThread = new Thread(SummatorCall);
									summatorThread.IsBackground = true;
									summatorThread.Start(array);

									for (int d = 0; d < neutrons.Length; d++)									
										neutrons[d] = new List<int>();									

									speed_x = (float)(spec_time / sw.Elapsed.TotalSeconds);
									speed_mbs = (float)(buf.Length / sw2.Elapsed.TotalSeconds / 1000000.0);

									Console.WriteLine(
										"save: {0,5}  time: {1,6:f2}  frame: {2,6}  speed: {3,6:f2}x  neutronsCount: {4}  neutronsDelta: {5}", 
										save_count, spec_time, frame, speed_x, neutronsCount, neutronsDelta);
								}
								break;
						}


						if (hi < detectors.Length && detectorsHashSet.Contains(hi))//detectors.Contains(hi))
						{
							long t = (long)lo | ((long)f4) << 24;
							if (t > fbeg && fbeg > fend)
							{
								float tmks = (float)((t - fbeg) * 16e-3);
								int tch = (int)(tmks / tau * kt[hi]) - channel0;
								if (tch >= 0 && tch < channelsCount)
								{
									neutrons[hi].Add(tch);
									neutronsCount++;
								}						
							}
						}
					}
				}
			}
		}
    }
}
