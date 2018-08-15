namespace GGM_MultiGPU
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            this.checkBox2 = new System.Windows.Forms.CheckBox();
            this.checkBox3 = new System.Windows.Forms.CheckBox();
            this.checkBox4 = new System.Windows.Forms.CheckBox();
            this.checkBox5 = new System.Windows.Forms.CheckBox();
            this.checkBox6 = new System.Windows.Forms.CheckBox();
            this.checkBox7 = new System.Windows.Forms.CheckBox();
            this.label5 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.g0 = new System.Windows.Forms.Label();
            this.g1 = new System.Windows.Forms.Label();
            this.g2 = new System.Windows.Forms.Label();
            this.g3 = new System.Windows.Forms.Label();
            this.g4 = new System.Windows.Forms.Label();
            this.g5 = new System.Windows.Forms.Label();
            this.g6 = new System.Windows.Forms.Label();
            this.txtServer = new System.Windows.Forms.TextBox();
            this.txtLogin = new System.Windows.Forms.TextBox();
            this.txtPort = new System.Windows.Forms.TextBox();
            this.txtPwd = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(77, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Stratum Server";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(358, 31);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(92, 13);
            this.label2.TabIndex = 1;
            this.label2.Text = "Stratum Password";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(358, 8);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(65, 13);
            this.label3.TabIndex = 2;
            this.label3.Text = "Stratum Port";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(12, 31);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(72, 13);
            this.label4.TabIndex = 3;
            this.label4.Text = "Stratum Login";
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Enabled = false;
            this.checkBox1.Location = new System.Drawing.Point(15, 85);
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.Size = new System.Drawing.Size(102, 17);
            this.checkBox1.TabIndex = 4;
            this.checkBox1.Text = "CUDA Device 0";
            this.checkBox1.UseVisualStyleBackColor = true;
            this.checkBox1.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // checkBox2
            // 
            this.checkBox2.AutoSize = true;
            this.checkBox2.Enabled = false;
            this.checkBox2.Location = new System.Drawing.Point(173, 85);
            this.checkBox2.Name = "checkBox2";
            this.checkBox2.Size = new System.Drawing.Size(102, 17);
            this.checkBox2.TabIndex = 5;
            this.checkBox2.Text = "CUDA Device 1";
            this.checkBox2.UseVisualStyleBackColor = true;
            this.checkBox2.CheckedChanged += new System.EventHandler(this.checkBox2_CheckedChanged);
            // 
            // checkBox3
            // 
            this.checkBox3.AutoSize = true;
            this.checkBox3.Enabled = false;
            this.checkBox3.Location = new System.Drawing.Point(361, 85);
            this.checkBox3.Name = "checkBox3";
            this.checkBox3.Size = new System.Drawing.Size(102, 17);
            this.checkBox3.TabIndex = 6;
            this.checkBox3.Text = "CUDA Device 2";
            this.checkBox3.UseVisualStyleBackColor = true;
            this.checkBox3.CheckedChanged += new System.EventHandler(this.checkBox3_CheckedChanged);
            // 
            // checkBox4
            // 
            this.checkBox4.AutoSize = true;
            this.checkBox4.Enabled = false;
            this.checkBox4.Location = new System.Drawing.Point(575, 85);
            this.checkBox4.Name = "checkBox4";
            this.checkBox4.Size = new System.Drawing.Size(102, 17);
            this.checkBox4.TabIndex = 7;
            this.checkBox4.Text = "CUDA Device 3";
            this.checkBox4.UseVisualStyleBackColor = true;
            this.checkBox4.CheckedChanged += new System.EventHandler(this.checkBox4_CheckedChanged);
            // 
            // checkBox5
            // 
            this.checkBox5.AutoSize = true;
            this.checkBox5.Enabled = false;
            this.checkBox5.Location = new System.Drawing.Point(15, 135);
            this.checkBox5.Name = "checkBox5";
            this.checkBox5.Size = new System.Drawing.Size(102, 17);
            this.checkBox5.TabIndex = 8;
            this.checkBox5.Text = "CUDA Device 4";
            this.checkBox5.UseVisualStyleBackColor = true;
            this.checkBox5.CheckedChanged += new System.EventHandler(this.checkBox5_CheckedChanged);
            // 
            // checkBox6
            // 
            this.checkBox6.AutoSize = true;
            this.checkBox6.Enabled = false;
            this.checkBox6.Location = new System.Drawing.Point(173, 135);
            this.checkBox6.Name = "checkBox6";
            this.checkBox6.Size = new System.Drawing.Size(102, 17);
            this.checkBox6.TabIndex = 9;
            this.checkBox6.Text = "CUDA Device 5";
            this.checkBox6.UseVisualStyleBackColor = true;
            this.checkBox6.CheckedChanged += new System.EventHandler(this.checkBox6_CheckedChanged);
            // 
            // checkBox7
            // 
            this.checkBox7.AutoSize = true;
            this.checkBox7.Enabled = false;
            this.checkBox7.Location = new System.Drawing.Point(361, 135);
            this.checkBox7.Name = "checkBox7";
            this.checkBox7.Size = new System.Drawing.Size(102, 17);
            this.checkBox7.TabIndex = 10;
            this.checkBox7.Text = "CUDA Device 6";
            this.checkBox7.UseVisualStyleBackColor = true;
            this.checkBox7.CheckedChanged += new System.EventHandler(this.checkBox7_CheckedChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(12, 62);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(103, 13);
            this.label5.TabIndex = 11;
            this.label5.Text = "Select active GPUs:";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(15, 197);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(166, 41);
            this.button1.TabIndex = 12;
            this.button1.Text = "Start All";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // g0
            // 
            this.g0.AutoSize = true;
            this.g0.Location = new System.Drawing.Point(12, 105);
            this.g0.Name = "g0";
            this.g0.Size = new System.Drawing.Size(39, 13);
            this.g0.TabIndex = 13;
            this.g0.Text = "GPU 0";
            // 
            // g1
            // 
            this.g1.AutoSize = true;
            this.g1.Location = new System.Drawing.Point(170, 105);
            this.g1.Name = "g1";
            this.g1.Size = new System.Drawing.Size(39, 13);
            this.g1.TabIndex = 14;
            this.g1.Text = "GPU 1";
            // 
            // g2
            // 
            this.g2.AutoSize = true;
            this.g2.Location = new System.Drawing.Point(358, 105);
            this.g2.Name = "g2";
            this.g2.Size = new System.Drawing.Size(39, 13);
            this.g2.TabIndex = 15;
            this.g2.Text = "GPU 2";
            // 
            // g3
            // 
            this.g3.AutoSize = true;
            this.g3.Location = new System.Drawing.Point(572, 105);
            this.g3.Name = "g3";
            this.g3.Size = new System.Drawing.Size(39, 13);
            this.g3.TabIndex = 16;
            this.g3.Text = "GPU 3";
            // 
            // g4
            // 
            this.g4.AutoSize = true;
            this.g4.Location = new System.Drawing.Point(12, 155);
            this.g4.Name = "g4";
            this.g4.Size = new System.Drawing.Size(39, 13);
            this.g4.TabIndex = 17;
            this.g4.Text = "GPU 4";
            // 
            // g5
            // 
            this.g5.AutoSize = true;
            this.g5.Location = new System.Drawing.Point(170, 155);
            this.g5.Name = "g5";
            this.g5.Size = new System.Drawing.Size(39, 13);
            this.g5.TabIndex = 18;
            this.g5.Text = "GPU 5";
            // 
            // g6
            // 
            this.g6.AutoSize = true;
            this.g6.Location = new System.Drawing.Point(358, 155);
            this.g6.Name = "g6";
            this.g6.Size = new System.Drawing.Size(39, 13);
            this.g6.TabIndex = 19;
            this.g6.Text = "GPU 6";
            // 
            // txtServer
            // 
            this.txtServer.Location = new System.Drawing.Point(95, 6);
            this.txtServer.Name = "txtServer";
            this.txtServer.Size = new System.Drawing.Size(236, 20);
            this.txtServer.TabIndex = 20;
            this.txtServer.TextChanged += new System.EventHandler(this.txtServer_TextChanged);
            // 
            // txtLogin
            // 
            this.txtLogin.Location = new System.Drawing.Point(95, 28);
            this.txtLogin.Name = "txtLogin";
            this.txtLogin.Size = new System.Drawing.Size(236, 20);
            this.txtLogin.TabIndex = 21;
            this.txtLogin.TextChanged += new System.EventHandler(this.txtLogin_TextChanged);
            // 
            // txtPort
            // 
            this.txtPort.Location = new System.Drawing.Point(470, 5);
            this.txtPort.Name = "txtPort";
            this.txtPort.Size = new System.Drawing.Size(96, 20);
            this.txtPort.TabIndex = 22;
            this.txtPort.TextChanged += new System.EventHandler(this.txtPort_TextChanged);
            // 
            // txtPwd
            // 
            this.txtPwd.Location = new System.Drawing.Point(470, 28);
            this.txtPwd.Name = "txtPwd";
            this.txtPwd.Size = new System.Drawing.Size(309, 20);
            this.txtPwd.TabIndex = 23;
            this.txtPwd.TextChanged += new System.EventHandler(this.txtPwd_TextChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(791, 250);
            this.Controls.Add(this.txtPwd);
            this.Controls.Add(this.txtPort);
            this.Controls.Add(this.txtLogin);
            this.Controls.Add(this.txtServer);
            this.Controls.Add(this.g6);
            this.Controls.Add(this.g5);
            this.Controls.Add(this.g4);
            this.Controls.Add(this.g3);
            this.Controls.Add(this.g2);
            this.Controls.Add(this.g1);
            this.Controls.Add(this.g0);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.checkBox7);
            this.Controls.Add(this.checkBox6);
            this.Controls.Add(this.checkBox5);
            this.Controls.Add(this.checkBox4);
            this.Controls.Add(this.checkBox3);
            this.Controls.Add(this.checkBox2);
            this.Controls.Add(this.checkBox1);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.MaximizeBox = false;
            this.Name = "Form1";
            this.ShowIcon = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "GGM CUDA Multi GPU Launcher";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.CheckBox checkBox2;
        private System.Windows.Forms.CheckBox checkBox3;
        private System.Windows.Forms.CheckBox checkBox4;
        private System.Windows.Forms.CheckBox checkBox5;
        private System.Windows.Forms.CheckBox checkBox6;
        private System.Windows.Forms.CheckBox checkBox7;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label g0;
        private System.Windows.Forms.Label g1;
        private System.Windows.Forms.Label g2;
        private System.Windows.Forms.Label g3;
        private System.Windows.Forms.Label g4;
        private System.Windows.Forms.Label g5;
        private System.Windows.Forms.Label g6;
        private System.Windows.Forms.TextBox txtServer;
        private System.Windows.Forms.TextBox txtLogin;
        private System.Windows.Forms.TextBox txtPort;
        private System.Windows.Forms.TextBox txtPwd;
    }
}

