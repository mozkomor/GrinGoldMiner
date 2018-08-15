using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GGM_MultiGPU
{
    public partial class Form1 : Form
    {
        List<Process> miners = new List<Process>();

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            button1.Enabled = false;

            for (int i = 0; i < 7; i++)
            {
                CheckBox cb = checkBox1;
                Label lbl = g0;

                switch (i)
                {
                    case 0:
                        cb = checkBox1;
                        lbl = g0;
                        break;
                    case 1:
                        cb = checkBox2;
                        lbl = g1;
                        break;
                    case 2:
                        cb = checkBox3;
                        lbl = g2;
                        break;
                    case 3:
                        cb = checkBox4;
                        lbl = g3;
                        break;
                    case 4:
                        cb = checkBox5;
                        lbl = g4;
                        break;
                    case 5:
                        cb = checkBox6;
                        lbl = g5;
                        break;
                    case 6:
                        cb = checkBox7;
                        lbl = g6;
                        break;
                }

                if (cb.Enabled && cb.Checked)
                {
                    try
                    {
                        string lm = lbl.Text.Contains("1080 Ti") ? "" : "-lm";
                        string poolLogin = txtLogin.Text != "" ? "-l "+txtLogin.Text : "";
                        string poolPwd = txtPwd.Text != "" ? "-p " + txtPwd.Text : "";
                        string arg = string.Format("Theta.dll -d {0} -a {1}:{2} {3} {4} {5}", i, txtServer.Text, txtPort.Text, lm, poolLogin, poolPwd);

                        Process p = Process.Start(new ProcessStartInfo()
                        {
                            FileName =  "dotnet",
                            Arguments = arg
                        });

                        miners.Add(p);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.Message);
                    }
                }
            }

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                txtLogin.Text = Properties.Settings.Default.login;
                txtServer.Text = Properties.Settings.Default.server;
                txtPort.Text = Properties.Settings.Default.port;
                txtPwd.Text = Properties.Settings.Default.password;

                checkBox1.Checked = Properties.Settings.Default.GPU0;
                checkBox2.Checked = Properties.Settings.Default.GPU1;
                checkBox3.Checked = Properties.Settings.Default.GPU2;
                checkBox4.Checked = Properties.Settings.Default.GPU3;
                checkBox5.Checked = Properties.Settings.Default.GPU4;
                checkBox6.Checked = Properties.Settings.Default.GPU5;
                checkBox7.Checked = Properties.Settings.Default.GPU6;
            }
            catch
            {

            }

            try
            {
                int cnt = ManagedCuda.CudaContext.GetDeviceCount();

                for (int i = 0; i < cnt; i++)
                {
                    var name = ManagedCuda.CudaContext.GetDeviceName(i);
                    var info = ManagedCuda.CudaContext.GetDeviceInfo(i);

                    var mem = info.TotalGlobalMemory;

                    SetDevice(i, name, info, mem);
                }
            }
            catch
            {

            }
        }

        private void SetDevice(int i, string name, CudaDeviceProperties info, SizeT mem)
        {
            CheckBox cb = checkBox1;
            Label lbl = g0;

            switch (i)
            {
                case 0:
                    cb = checkBox1;
                    lbl = g0;
                    break;
                case 1:
                    cb = checkBox2;
                    lbl = g1;
                    break;
                case 2:
                    cb = checkBox3;
                    lbl = g2;
                    break;
                case 3:
                    cb = checkBox4;
                    lbl = g3;
                    break;
                case 4:
                    cb = checkBox5;
                    lbl = g4;
                    break;
                case 5:
                    cb = checkBox6;
                    lbl = g5;
                    break;
                case 6:
                    cb = checkBox7;
                    lbl = g6;
                    break;
            }

            cb.Enabled = true;
            lbl.Text = name;
            lbl.Tag = mem;
        }

        private void txtServer_TextChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.server = txtServer.Text;
            Properties.Settings.Default.Save();
        }

        private void txtLogin_TextChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.login = txtLogin.Text;
            Properties.Settings.Default.Save();
        }

        private void txtPort_TextChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.port = txtPort.Text;
            Properties.Settings.Default.Save();
        }

        private void txtPwd_TextChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.password = txtPwd.Text;
            Properties.Settings.Default.Save();
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU0 = checkBox1.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU1 = checkBox2.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU2 = checkBox3.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox4_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU3 = checkBox4.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox5_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU4 = checkBox5.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox6_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU5 = checkBox6.Checked;
            Properties.Settings.Default.Save();
        }

        private void checkBox7_CheckedChanged(object sender, EventArgs e)
        {
            Properties.Settings.Default.GPU6 = checkBox7.Checked;
            Properties.Settings.Default.Save();
        }
    }
}
