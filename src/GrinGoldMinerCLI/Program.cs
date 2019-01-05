using System;
using System.IO;
using System.Threading.Tasks;
using Mozkomor.GrinGoldMiner;

namespace Mozkomor.GrinGoldMinerCLI
{
    class Program
    {
        private static volatile bool IsTerminated;

        static void Main(string[] args)
        {
            Console.CancelKeyPress += delegate
            {
                Console.WriteLine("Ctrl+C - Exitting");
                Close();
            };

            if (DateTime.Today >= new DateTime(2019, 1, 14))
            {
                Console.WriteLine("!!! This version of GrinGoldMiner is outdated. Please go to https://github.com/mozkomor/GrinGoldMiner/releases and downlaod the latest release.");
                Logger.Log(LogLevel.ERROR, "!!! This version of GrinGoldMiner is outdated. Please go to https://github.com/mozkomor/GrinGoldMiner/releases and downlaod the latest release.");
                Console.ReadLine();
                Close();
            }

            var dir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var configPath = Path.Combine(dir, "config.xml");
            Config config = new Config();
            if (File.Exists(configPath))
            {
                config = Serialization.DeSerialize<Config>(configPath);
            }
            else
            {
                var generated_config = Config.GetDefaultConfig();
                generated_config.GPUOptions.Clear();
                // CUDA
                try
                {
                    Worker nv = new Worker(SharedSerialization.WorkerType.NVIDIA, 0, 0);
                    var devices = nv.GetDevices();

                    foreach (var dev in devices)
                    {
                        if (dev.memory > (long)1024 * 1024 * 1024 * 7)
                        {
                            generated_config.GPUOptions.Add(new SharedSerialization.GPUOption()
                            {
                                GPUName = dev.name,
                                DeviceID = dev.deviceID,
                                Enabled = true,
                                GPUType = SharedSerialization.WorkerType.NVIDIA,
                                PlatformID = 0
                            });
                        }
                    }
                }
                catch { }

                Task.Delay(2000).Wait();

                // AMD
                try
                {
                    Worker amd = new Worker(SharedSerialization.WorkerType.AMD, 0, 0);
                    var devices = amd.GetDevices();

                    foreach (var dev in devices)
                    {
                        if ((dev.memory > (long)1024 * 1024 * 1024 * 3) && dev.platformName.ToLower().Contains("amd"))
                        {
                            generated_config.GPUOptions.Add(new SharedSerialization.GPUOption()
                            {
                                GPUName = dev.name,
                                DeviceID = dev.deviceID,
                                Enabled = true,
                                GPUType = SharedSerialization.WorkerType.AMD,
                                PlatformID = dev.platformID
                            });
                        }
                    }
                }
                catch
                {
                    ;
                }

                Serialization.Serialize<Config>(generated_config, configPath);
                Console.WriteLine($"ERROR: missing config.xml, created new config.xml in directory with miner ({configPath}), please set the values in this file");
                Console.ReadLine();
                Close();
            }

            Logger.SetLogOptions(config.LogOptions);
            WorkerManager.Init(config);
            ConnectionManager.Init(config);

            Task.Factory.StartNew(() => { WriteGUI(); }, TaskCreationOptions.LongRunning);

            while (Console.ReadKey().Key != ConsoleKey.Q)
            {
            }
            Close();


            Console.WriteLine();
        }

        public static void Close()
        {
            try
            {
                IsTerminated = true;
                ConnectionManager.CloseAll();
                Environment.Exit(0);
            }
            catch { }
        }

        private static void WriteGUI()
        {
            while (!IsTerminated)
            {
                //Console.Clear();
                Console.CursorVisible = false;
                Console.SetCursorPosition(0, 0);

                Console.WriteLine("Grin Gold Miner 2.0");
                Console.WriteLine("------------------------------------------------------------------------------------------");
                WipeLine();
                Console.Write("Mining for: ");
                Console.CursorLeft = 20;
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("USER"); Console.ResetColor();
                Console.CursorLeft = 35;
                Console.WriteLine("Stratum Server: stratum.pool.com");
                WipeLine();
                Console.Write("Connection status: ");
                Console.CursorLeft = 20;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("CONNECTED"); Console.ResetColor();

                Console.CursorLeft = 35;
                Console.WriteLine("Last job in:    35 seconds");
                WipeLine();
                Console.Write("Submitted shares: ");
                Console.CursorLeft = 20;
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("578"); Console.ResetColor();

                Console.CursorLeft = 35;
                Console.WriteLine("Last share:     10 seconds");
                WipeLine();
                Console.WriteLine("------------------------------------------------------------------------------------------");
                WipeLine();
                foreach (var w in WorkerManager.workers)
                {
                    w.PrintStatusLinesToConsole();
                }
                Console.WriteLine("------------------------------------------------------------------------------------------");
                //Console.ForegroundColor = ConsoleColor.Yellow;
                //Console.WriteLine("Last log messages:"); Console.ResetColor();
                WipeLines(5);
                Console.WriteLine(Logger.GetlastLogs());
                WipeLine();

                Task.Delay(500).Wait();
            }
        }

        private static void WipeLine()
        {
            Console.Write("                                                                                                              ");
            Console.CursorLeft = 0;
        }
        private static void WipeLines(int cnt)
        {
            for (int i = 0; i < cnt; i++)
            {
                Console.WriteLine("                                                                                                              ");
                Console.CursorLeft = 0;
            }
            Console.CursorTop -= cnt;
        }
    }
}
