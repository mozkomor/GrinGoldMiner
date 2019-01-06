using System;
using System.IO;
using System.Threading.Tasks;
using Mozkomor.GrinGoldMiner;

namespace Mozkomor.GrinGoldMinerCLI
{
    class Program
    {
        private static volatile bool IsTerminated;
        public static Config config;

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
            config = new Config();
            if (File.Exists(configPath))
            {
                // try catch this !!
                config = Serialization.DeSerialize<Config>(configPath);
            }
            else
            {
                var generated_config = Config.GetDefaultConfig();
                generated_config.GPUOptions.Clear();

                Console.WriteLine("No config file found, it will be generated now....");
                Console.WriteLine("Autodetecting GPUs...");
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

                Console.WriteLine($"Detected {generated_config.GPUOptions.Count} suitable nvidia devices");
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

                    Console.WriteLine($"Detected {devices.Count} suitable AMD devices");
                }
                catch
                {
                    ;
                }

                Console.WriteLine($"Enter your mining pool stratum address:");
                string pool = Console.ReadLine();
                generated_config.PrimaryConnection.ConnectionAddress = pool;
                string port = "13416";
                if (pool.Contains(":"))
                {
                    // get port from url
                    try
                    {
                        generated_config.PrimaryConnection.ConnectionAddress = pool.Split(':')[0];
                        port = pool.Split(':')[1];
                    }
                    catch { }
                }
                else
                {
                    Console.WriteLine($"Enter your mining pool stratum port:");
                    port = Console.ReadLine();
                }
                try
                {
                    generated_config.PrimaryConnection.ConnectionPort = int.Parse(port.Trim());
                }
                catch
                {
                    Console.WriteLine($"Unable to parse port, please edit the config manually.");
                }                

                Console.WriteLine($"Use TLS? (y/n)         [Note that the pool:port must support TLS]");
                string ssl = Console.ReadLine();
                generated_config.PrimaryConnection.Ssl = ssl == "y" || ssl == "Y" || ssl == "YES" || ssl == "yes";
                Console.WriteLine($"Enter your pool login:");
                generated_config.PrimaryConnection.Login = Console.ReadLine();
                Console.WriteLine($"Enter your pool password:");
                generated_config.PrimaryConnection.Password = Console.ReadLine();

                Serialization.Serialize<Config>(generated_config, configPath);
                Console.WriteLine();
                Console.WriteLine($"Created new config.xml in executable directory ({configPath}), edit this file to change miner settings.");
                Console.ReadLine();
                Close();
                return;
            }

            Task.Factory.StartNew(() => { WriteGUI(); }, TaskCreationOptions.LongRunning);

            Logger.SetLogOptions(config.LogOptions);
            WorkerManager.Init(config);


            // this blocks on no connection ?
            ConnectionManager.Init(config);

            ConsoleKey kk;
            while ((kk = Console.ReadKey().Key) != ConsoleKey.Q)
            {
                Console.Clear();

                switch (kk)
                {
                    case ConsoleKey.L:
                        // show full log flow
                        if (Logger.consoleMode == ConsoleOutputMode.STATIC_TUI)
                        {
                            Logger.consoleMode = ConsoleOutputMode.ROLLING_LOG;
                            Console.WriteLine("Rolling log mode: enabled");
                        }
                        else
                            Logger.consoleMode = ConsoleOutputMode.STATIC_TUI;
                        break;
                }
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
            }
            catch { }
            finally
            {
                Environment.Exit(0);
            }
        }

        private static void WriteGUI()
        {
            Console.Clear();
            while (!IsTerminated)
            {
                switch (Logger.consoleMode)
                {
                    case ConsoleOutputMode.STATIC_TUI:
                        {
                            if (DateTime.Now.Second == 0) Console.Clear();
                            Console.CursorVisible = false;
                            Console.SetCursorPosition(0, 0);
                            WipeLine();

                            var conn = ConnectionManager.curr;

                            string remote = "---";
                            string who = "USER";
                            string status = "CONNECTING";

                            if (conn != null)
                            {
                                remote = $"{conn.ip}:{conn.port}";
                                if (config != null && 
                                    (config.PrimaryConnection != null && config.PrimaryConnection.ConnectionAddress.Trim() == conn.ip) ||
                                    (config.SecondaryConnection != null && config.SecondaryConnection.ConnectionAddress.Trim() == conn.ip)
                                    )
                                    who = "USER";
                                else
                                    who = "FEE";

                                status = conn.IsConnected ? "CONNECTED" : "DISCONNECTED";
                            }

                            Console.WriteLine("Grin Gold Miner 2.0");
                            Console.WriteLine("------------------------------------------------------------------------------------------");
                            WipeLine();
                            Console.Write("Mining for: ");
                            Console.CursorLeft = 20;
                            Console.ForegroundColor = who == "USER" ? ConsoleColor.Cyan : ConsoleColor.Blue;
                            Console.Write(who); Console.ResetColor(); //TODO
                            Console.CursorLeft = 35;
                            Console.WriteLine($"Stratum Server: {remote}"); //TODO
                            WipeLine();
                            Console.Write("Connection status: ");
                            Console.CursorLeft = 20;
                            Console.ForegroundColor = status == "CONNECTED" ? ConsoleColor.Green : (status == "CONNECTING" ? ConsoleColor.Yellow : ConsoleColor.Red);
                            Console.Write(status); Console.ResetColor(); //TODO

                            Console.CursorLeft = 35;
                            Console.WriteLine($"Last job in:    {(DateTime.Now-WorkerManager.lastJob).TotalSeconds:F0} seconds");
                            WipeLine();
                            Console.Write("Submitted shares: ");
                            Console.CursorLeft = 20;
                            Console.ForegroundColor = ConsoleColor.Yellow;
                            Console.Write($"{StratumConnet.totalShares}"); Console.ResetColor();

                            Console.CursorLeft = 35;
                            Console.WriteLine($"Last share:     {(DateTime.Now-StratumConnet.lastShare).TotalSeconds:F0} seconds");
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
                        break;
                }
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
