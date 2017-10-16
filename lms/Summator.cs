using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    abstract class Summator
    {		
		public static int[][] spectrum; //detector - channel_value
		public static int channelsCount, channelWidth, strob;
		public static int[] detectors;
        public bool isSavingNow = false;
        public Action PreSaveAction = () => { };
        public Summator(int chcount, int chwidth, int[] dets, int strob_value)
        {
			channelsCount = chcount;
			channelWidth = chwidth;
			detectors = dets;
			strob = strob_value;
            spectrum = createSpectrumArray();
        }        

        protected int[][] createSpectrumArray()
        {
            int[][] spectr = new int[detectors.Length][];
            for (int i = 0; i < spectr.Length; i++)
                spectr[i] = new int[channelsCount];
            return spectr;
        }

		virtual public void AddValues(int[][] neutrons) //detector - neutron_channel
        {

        }  

        virtual public int[][] CalcFrame(int[][] frame)
        {
            return createSpectrumArray();
        }
        
        virtual public int[][] GetSpectrum()
        {
            return spectrum;
        }

        public void ClearSpectrum()
        {            
            for (int i = 0; i < spectrum.Length; i++)
                Parallel.For(0, spectrum[i].Length, index =>
                {
                    spectrum[i][index] = 0;
                });                
        }
        
        public void SaveSpectrum(string folder, int num)
        {
            int[][] spectr = GetSpectrum();
            SaveSpectrum(folder, num, spectr);
        }

        public void SaveSpectrum(string folder, int num, int[][] spectr)
        {            
            int[] ss = new int[channelsCount];
            foreach (int j in detectors) //=0; j<max_det; j++)
            {
                string spname = 
                    folder 
                    + "\\" + "sp_" + j.ToString("d2")
                    + "." + num.ToString("d3");
                int[] s = spectr[j];                
                BinaryWriter bw = new BinaryWriter(File.OpenWrite(spname));
                for (int i = 0; i < s.Length; i++)
                {
                    bw.Write(s[i]);
                    ss[i] += s[i];                    
                }
                bw.Close();

            }

            string spname2 = folder + "\\" + "sp_sum" /*+ "_" + time.ToString("f3")*/ + "." + num.ToString("d3");
            BinaryWriter bww = new BinaryWriter(File.OpenWrite(spname2));
            for (int i = 0; i < ss.Length; i++)
            {
                bww.Write(ss[i]);
            }
            bww.Close();

            //Console.WriteLine("Result saved to {0}", Environment.CurrentDirectory + "\\" + folder + "\\");
            isSavingNow = false;
        }
    }
}
