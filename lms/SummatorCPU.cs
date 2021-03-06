﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lms
{
    public class SummatorCPU : Summator
    {
		/*public static double[][] channels; //detector - channel_value
		public static int channelsCount, channelWidth, strob;
		public static int[] detectors;*/
        public SummatorCPU(int channelsCount, int channelWidth, int[] detectors, int strob) 
            : base(channelsCount, channelWidth, detectors, strob)
        {
            Console.WriteLine("__SummatorCPU__");
        }

        public override int[][] GetSpectrum()
        {
            int[][] output = spectrum.ToArray();
            spectrum = createSpectrumArray();
            return output;
        }

        public override void ClearSpectrum()
        {
            for (int i = 0; i < spectrum.Length; i++)
                Parallel.For(0, spectrum[i].Length, index =>
                {
                    spectrum[i][index] = 0;
                });
        }

        
        public /*override*/ int[][] CalcFrame2d2(int[][] frame)
        {
            int[][] spectr = createSpectrumArray();
            //for (int detector = 0; detector < frame.Length; detector++)
            Parallel.For(0, frame.Length, detector =>
            {
                int[] s = frame[detector];
                for (int i = 0; i < s.Length; i++)
                {
                    int ch = s[i];
                    int k1 = ch - strob; if (k1 < 0) k1 = 0;
                    int k2 = ch + strob; if (k2 > channelsCount - 1) k2 = channelsCount - 1;
                    for (int k = k1; k < k2; k++) spectr[detector][k] += 1;
                }
            });
            return spectr;
        }

        public override int[][] CalcFrame2d(int[][] frame)
        {
            int[][] spectr = createSpectrumArray();            
            int[][] channels = calcChannels(frame);
            
            for (int detector = 0; detector < frame.Length; detector++)                
                Parallel.For(0, channelsCount, ch =>
                {
                    int k1 = ch - strob; if (k1 < 0) k1 = 0;
                    int k2 = ch + strob; if (k2 > channelsCount - 1) k2 = channelsCount - 1;
                    int sum = 0;
                    for (int k = k1; k < k2; k++)
                        sum += channels[detector][k];
                    spectr[detector][ch] = sum;
                });            

            int[][] spectr2 = CalcFrame2d2(frame);

            int[][] difference = new int[spectr2.Length][];
            for (int i = 0; i < difference.Length; i++)
            {
                int[] diff = new int[spectr2[i].Length];
                for (int index = 0; index < diff.Length; index++)
                    diff[index] = spectr2[i][index] - spectr[i][index];
                difference[i] = diff;
            }

            return spectr;
        }

        public int[][] calcChannels(int[][] frame)
        {
            int[][] channels = createSpectrumArray();
            Parallel.For(0, frame.Length, detector =>
            {
                int[] s = frame[detector];
                for (int i = 0; i < s.Length; i++)
                {
                    int ch = s[i];
                    channels[detector][ch] += 1;
                }
            });
            return channels;
        }

        public int[,] calcChannelsJagged(int[][] frame)
        {
            int[,] channels = new int[frame.Length, channelsCount];
            Parallel.For(0, frame.Length, detector =>
            {
                int[] s = frame[detector];
                for (int i = 0; i < s.Length; i++)
                {
                    int ch = s[i];
                    channels[detector,ch] += 1;
                }
            });
            return channels;
        }

        public override void AddValues(int[][] neutrons)
        {            
            for (int detector = 0; detector < neutrons.Length; detector++)            
            {
                /*int[] s = neutrons[detector];
                Parallel.For(0, s.Length, i =>
                {
                    int ch = s[i];
                    int k1 = ch - strob; if (k1 < 0) k1 = 0;
                    int k2 = ch + strob; if (k2 > channelsCount - 1) k2 = channelsCount - 1;
                    for (int k = k1; k < k2; k++) spectrum[detector][k] += 1;
                });*/
            }
        }
    }
}
