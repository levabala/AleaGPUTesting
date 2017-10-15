using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace lms
{
    class Program
	{
		#region variables
		static int ref_ch0 = 0;
		static int ref_chan = 31600;
		static int ref_strob = 200;
		static int strob = 20;
		static int ref_det = 0;
		static List<int> dets = new List<int>();
		static double ref_phase = 0.7;
		static double ref_k = 0.7128;
		static double ref_r = 1 / ref_k;
		static int ref_tau = 1;
		static string ref_out = "low";
		static int ref_frames = 50;
		static List<string> names = new List<string>() { "*_raw.0??" };
		static List<string> namelist = new List<string>();
		static int max_det = 32;
		static int max_mks = 32768;
		static int maxch = max_mks / ref_tau; //32000;
		static int ref_delay1 = 3200;
		static bool save_all = false;
		static bool save_sum = false;
		static bool save_ssum = false;

		static List<int[]> sp = new List<int[]>(max_det);
		static double[] kt = new double[]
		{
			1,
			1,
			0.9963,
			0.9968,
			0.9942,
			0.9944,
			1,
			1,
			0.9970,
			0.9976,
			0.9944,
			0.9946,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1
		};
		#endregion		

		static Summator summator;

		static void Main(string[] args)
        {			
			init(args);

			ParameterizedThreadStart summatorThreadStart = new ParameterizedThreadStart(SummatorCall);

			summator = new SummatorCPU(ref_chan, max_mks, dets.ToArray(), strob);
			Parser.Parse(namelist, strob, ref_chan, max_mks, ref_frames, ref_tau, kt, dets.ToArray(), ref_ch0, summatorThreadStart);

			/*Visualizer visual = new Visualizer();			
			visual.DrawSpec(summator.channels[0]);
			visual.Show();*/

            Console.ReadKey();
        }

		public static void SummatorCall(object arg)
		{
			int[][] neutrons = arg as int[][];
			summator.AddValues(neutrons);			
		}

		public static void init(string[] args)
		{
			string help_msg =
			"List Mode Transformer by db' 2017-10-14\n\n" +
			"usage: lmt <params> \n\n" +
			"use single param as <filename> or <search_mask> for .raw files in current folder \n" +
			"like <zns6_raw.01> or <*_raw.???> \n\n" +
			"combination of following -keys can be used:\n\n" +
			"-raw <mask or name> - listmode files to load\n" +
			"-det <flag> - detector flag to calc (0x00-0xEF)\n" +
			"-dor - select detectors 0-11\n" +
			"-dpr - select detectors 12-31\n" +
			"-strob <mks> - strob width (800)\n" +
				//"-delay1 <mks> - start-signal 0xf0 to reactor-pulse delay (3300)\n" +
				//"-delay2 <mks> - first channel of -h file (1700)\n" +
			"-ch0 <number> - first channel to write (0) \n" +
			"-chan <number> - number of channels to write (7900)\n" +
				//"-tau <mks> - time-channel width of -h file (2)\n" +
				//"-h <filename> - load .h reference spectrum (blue)\n" +
				//"-cfg <filename> - load .config from file\n" +
			"-w <mks> - calculate with channel width tau = <mks> \n" +
			"-o <name> - output folder\n" +
			"-frames <number> - N frames to sum/write in low res mode\n" +
			"-all - save all calculated detectors\n" +
			"-sum - save only sum calculated\n" +
			"-ssum - save super sum\n"
			;

			//string[] args = Environment.GetCommandLineArgs();
			if (args.Length < 1)
			{
				Console.WriteLine(help_msg);
				return;
			}

			names.Clear();

			if (args.Length == 1) // one argument treating as single .raw or search mask
			{
				names.Add(args[1]);
			}

			if (args.Length > 1)
			{
				for (int i = 0; i < args.Length; i++)
				{
					switch (args[i])
					{
						case "-raw": names.Add(args[i + 1]); break;
						case "-det": if (int.TryParse(args[i + 1], out ref_det)) dets.Add(ref_det); break;
						case "-mks": int.TryParse(args[i + 1], out max_mks); break;
						//case "-hkl": ReadHKL(args[i + 1]); break;
						//case "-tau": int.TryParse(args[i + 1], out ref_tau); break;
						case "-delay1": int.TryParse(args[i + 1], out ref_delay1); break;
						//case "-delay2": if (int.TryParse(args[i + 1], out ref_delay2)) Delay2Mks = ref_delay2; break;
						//case "-chan": if (int.TryParse(args[i + 1], out ref_chan))
						//		SetupMaxMks(ref_chan); break;
						//case "-ch0": if (int.TryParse(args[i + 1], out ref_ch0))
						//		SetupMaxMks(ref_chan); break;
						case "-strob": int.TryParse(args[i + 1], out ref_strob); break;
						case "-phase": double.TryParse(args[i + 1], out ref_phase); break;
						case "-r": if (double.TryParse(args[i + 1], out ref_r)) ref_k = 1 / ref_r; break;
						case "-k": if (double.TryParse(args[i + 1], out ref_k)) ref_r = 1 / ref_k; break;
						//case "-h": LoadH(ref_hname = args[i + 1]); break;
						//case "-cfg": config_path = args[i + 1]; break;
						//case "-run": run = true; break;
						//case "-pos": if (int.TryParse(args[i + 1], out ref_pos)) c.invert = (ref_pos == 1 ? true : false); break;
						//case "-deb": c.debounce = true; break;
						//case "-w": int.TryParse(args[i + 1], out ref_tau); SetupMaxChan(max_mks); break;
						case "-o": ref_out = args[i + 1]; break;
						//case "-anal": anal = true; break;
						//case "-avg": avg = true; break;
						//case "-rpm": write_rpm = true; break;
						//case "-nolog": c.nolog = true; break;
						//case "-ff": if (double.TryParse(args[i + 1], out ref_ff)) c.ff = ref_ff; break;
						case "-frames": int.TryParse(args[i + 1], out ref_frames); break;
						//case "-low": c.lowRes = true; break;
						case "-dor": SelectDOR(); break;
						case "-dpr": SelectDPR(); break;
						case "-all": save_all = true; break;
						case "-sum": save_sum = true; break;
						case "-ssum": save_ssum = true; break;
					}
				}
			}

			maxch = max_mks / ref_tau;
			for (int d = 0; d < max_det; d++) sp.Add(new int[maxch]);
			strob = ref_strob / ref_tau;			

			namelist = new List<string>();

			foreach (string s in names)
			{
				if (s.Contains('*') || s.Contains('?'))
				{
					string[] ss = Directory.GetFiles(".", s);
					if (ss.Length > 0) namelist.AddRange(ss);
				}
				else
					if (File.Exists(s)) namelist.Add(s);
			}
			namelist.Sort();

			foreach (string fn in namelist)
			{
				Console.WriteLine(fn);
			}

			foreach (int d in dets)
			{
				Console.Write("{0} ", d);
			}
			Console.WriteLine();
			Console.WriteLine("w = {0} mks", ref_tau);
			Console.WriteLine("strob = {0} chan", ref_strob);
			Console.WriteLine("frames = {0}", ref_frames);
			Console.WriteLine("output = {0}", ref_out);

			//			Console.ReadKey();

			if (!Directory.Exists(ref_out)) Directory.CreateDirectory(ref_out);
		}

		static void SelectDOR()
		{
			dets.AddRange(Enumerable.Range(0, 12));
		}
		static void SelectDPR()
		{
			dets.AddRange(Enumerable.Range(12, 20));
		}

		static void RemoveDuplets(List<int> l)
		{
			l.Sort();
			for (int i = 1; i < l.Count; i++)
			{
				if (l[i] == l[i - 1]) l.RemoveAt(i);
			}
		}
    }
}
