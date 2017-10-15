using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace lms
{
	public partial class Visualizer : Form
	{
		public Visualizer()
		{
			InitializeComponent();
		}

		private void Visualizer_Load(object sender, EventArgs e)
		{
			DoubleBuffered = true;
		}
		public void DrawSpec(double[] sp)
		{
			Graphics g = CreateGraphics();
			g.Clear(Color.White);

			PointF[] pt = new PointF[sp.Length];

			for (int i = 0; i < sp.Length; i++)
				pt[i] = new PointF(i, ClientRectangle.Height - (float)sp[i]);

			g.DrawLines(Pens.Black, pt);			
		}
	}
}
