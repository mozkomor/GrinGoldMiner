// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using Terminal.Gui;

namespace GGM
{
    class Config
    {
        public const string GGMCFile = "config.xml";
        public static GGMConfig GGMC { get; set; }

        public static bool LoadConfig(string[] args)
        {
            try
            {
                var parser = new SimpleCommandLineParser();
                parser.Parse(args);

                bool exists = File.Exists(GGMCFile);

                if (!exists)
                {
                    GenerateConfig();
                }
                else
                    GGMC = DeSerializeString<GGMConfig>(File.ReadAllText(GGMCFile));

                return true;
            }
            catch (Exception ex)
            {
                Logger.Log(LogType.FatalError, "Error while loading config.", ex);
                return false;
            }
        }

        private static void GenerateConfig()
        {
            GGMC = new GGMConfig() { GPUs = new List<GPU>() };

            Application.Init();

            var top = Application.Top;

            // Creates the top-level window to show
            var win = new Window(new Rect(0, 1, top.Frame.Width, top.Frame.Height - 1), "GGM Config");
            top.Add(win);

            const int gpuOffset = 17;

            // Add some controls
            win.Add(

                    new Label(3, 1, "Stratum Server: "),
                    new TextField(20, 1, 40, ""),
                    new Label(65, 1, "Port: "),
                    new TextField(75, 1, 10, "13416"),
                    // 4
                    new Label(3, 3, "Stratum Login: "),
                    new TextField(20, 3, 40, ""),
                    new Label(65, 3, "Password: "),
                    new TextField(75, 3, 40, ""),
                    // 8
                    new Label(3 + gpuOffset, 6, "GPU 1: "),
                    new CheckBox(12 + gpuOffset, 6, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 6, new[] { "NVIDIA", "AMD" }) ,
                    new Label(40 + gpuOffset, 6, "Device ID: "),
                    new TextField(52 + gpuOffset, 6, 10, ""),

                    new Label(3 + gpuOffset, 9, "GPU 2: "),
                    new CheckBox(12 + gpuOffset, 9, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 9, new[] { "NVIDIA", "AMD" }),
                    new Label(40 + gpuOffset, 9, "Device ID: "),
                    new TextField(52 + gpuOffset, 9, 10, ""),

                    new Label(3 + gpuOffset, 12, "GPU 3: "),
                    new CheckBox(12 + gpuOffset, 12, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 12, new[] { "NVIDIA", "AMD" }),
                    new Label(40 + gpuOffset, 12, "Device ID: "),
                    new TextField(52 + gpuOffset, 12, 10, ""),

                    new Label(3 + gpuOffset, 15, "GPU 4: "),
                    new CheckBox(12 + gpuOffset, 15, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 15, new[] { "NVIDIA", "AMD" }),
                    new Label(40 + gpuOffset, 15, "Device ID: "),
                    new TextField(52 + gpuOffset, 15, 10, ""),

                    new Label(3 + gpuOffset, 18, "GPU 5: "),
                    new CheckBox(12 + gpuOffset, 18, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 18, new[] { "NVIDIA", "AMD" }),
                    new Label(40 + gpuOffset, 18, "Device ID: "),
                    new TextField(52 + gpuOffset, 18, 10, ""),

                    new Label(3 + gpuOffset, 21, "GPU 6: "),
                    new CheckBox(12 + gpuOffset, 21, "Enabled"),
                    new RadioGroup(26 + gpuOffset, 21, new[] { "NVIDIA", "AMD" }),
                    new Label(40 + gpuOffset, 21, "Device ID: "),
                    new TextField(52 + gpuOffset, 21, 10, ""),

                    new Button(3, 24, "Save", false) { Clicked = () => { Save(); } } 
            );

            Application.Top.Add(win);
            Application.Run();
        }

        private static void Save()
        {
            try
            {
                var win = Application.Top.Subviews[0].Subviews[0].Subviews;

                GGMC.DebugLogs = true;

                GGMC.StratumServer = (win[1] as TextField).Text.ToString();
                GGMC.StratumPort = int.Parse((win[3] as TextField).Text.ToString());
                GGMC.Login = (win[5] as TextField).Text.ToString();
                GGMC.Password = (win[7] as TextField).Text.ToString();

                int gid = 0;
                for (int g = 0; g < 6; g++)
                {
                    int start = g * 5 + 8;
                    bool enabled = (win[start + 1] as CheckBox).Checked;

                    if (enabled)
                    {
                        GPU gpu = new GPU() {
                            GPUID = gid++,
                            DeviceID = int.Parse((win[start + 4] as TextField).Text.ToString()),
                            Type = (win[start + 2] as RadioGroup).Selected == 0 ? GPUtype.CUDA : GPUtype.AMDCL
                        };
                        GGMC.GPUs.Add(gpu);
                    }
                }

                File.WriteAllText(GGMCFile, SerializeObject(GGMC));

                Application.Top.Running = false;
            }
            catch (Exception ex)
            {
                MessageBox.ErrorQuery(50, 5, "Error", ex.Message, "Ok");
                Logger.Log(LogType.FatalError, "Error while saving config.", ex);
            }
        }


        //http://blog.gauffin.org/2014/12/simple-command-line-parser/
        public class SimpleCommandLineParser
        {
            public SimpleCommandLineParser()
            {
                Arguments = new Dictionary<string, string[]>();
            }
            public IDictionary<string, string[]> Arguments { get; private set; }
            public void Parse(string[] args)
            {
                var currentName = "";
                var values = new List<string>();
                foreach (var arg in args)
                {
                    if (arg.StartsWith("-"))
                    {
                        if (currentName != "")
                            Arguments[currentName] = values.ToArray();
                        values.Clear();
                        currentName = arg.Substring(1);
                    }
                    else if (currentName == "")
                        Arguments[arg] = new string[0];
                    else
                        values.Add(arg);
                }
                if (currentName != "")
                    Arguments[currentName] = values.ToArray();
            }
            public bool Contains(string name)
            {
                return Arguments.ContainsKey(name);
            }
        }

        public static string SerializeObject<T>(T toSerialize)
        {
            XmlSerializer xmlSerializer = new XmlSerializer(toSerialize.GetType());

            using (StringWriter textWriter = new StringWriter())
            {
                xmlSerializer.Serialize(textWriter, toSerialize);
                return textWriter.ToString();
            }
        }

        public static T DeSerializeString<T>(string inputContent)
        {
            try
            {
                XmlSerializer s = new XmlSerializer(typeof(T));
                T newClass;
                using (TextReader r = new StringReader(inputContent))
                {
                    newClass = (T)s.Deserialize(r);
                }

                return newClass;
            }
            catch (Exception Ex)
            {
                return default(T);
            }
        }
    }

    public class GGMConfig
    {
        public string Login { get; set; }
        public string Password { get; set; }
        public string StratumServer { get; set; }
        public int StratumPort { get; set; }

        public bool DebugLogs { get; set; }

        public List<GPU> GPUs { get; set; }
    }

    public class GPU
    {
        public int GPUID { get; set; }
        public GPUtype Type { get; set; }
        public int DeviceID { get; set; }
        public string DeviceName { get; set; }
        public long DeviceMemory { get; set; }

        public float AverageGPS { get; set; }
        public int TotalGraphs { get; set; }
        public int TotalShares { get; set; }
        public int TotalSolutions { get; set; }

        [XmlIgnore]
        internal TrimDriver Driver { get; set; }
        [XmlIgnore]
        public string Status { get; set; }

        internal void Terminate()
        {
            if (Driver != null)
                Driver.Terminate();
        }
    }

    public enum GPUtype
    {
        CUDA,
        AMDCL
    }
}
