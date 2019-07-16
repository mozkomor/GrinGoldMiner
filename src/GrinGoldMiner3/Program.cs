using Mozkomor.GrinGoldMiner;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace GrinGoldMiner3
{
    class Program
    {
        private static volatile bool IsTerminated;
        public static volatile bool ChangeRemoteTerminate = false;//API received new config
        static Dictionary<string, string> cmdParams = new Dictionary<string, string>();
        public static readonly bool IsLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
        private static string configPath = "";
        public static Config config;

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            Console.WriteLine("Starting GrinGoldMiner 3.0 ...");

            Console.CancelKeyPress += delegate
            {
                Console.WriteLine("Ctrl+C - Exitting");
                Close();
            };

            #region parse args
            try
            {
                //https://stackoverflow.com/questions/16540640/how-to-pass-key-value-pairs-from-console-app-to-a-dictionary
                cmdParams = args.Select(a => a.Split(new[] { '=' }, 2))
                         .GroupBy(a => a[0], a => a.Length == 2 ? a[1] : null)
                         .ToDictionary(g => g.Key, g => g.FirstOrDefault());
            }
            catch
            {
                if (IsLinux)
                {
                    Console.WriteLine(@"ERROR PARSING ARGUMENTS. WILL CLOSE. Use args like this: ./GrinGoldMinerCLI configpath=/absolute/path/to/directory api-port=5777");
                }
                else
                {
                    Console.WriteLine(@"ERROR PARSING ARGUMENTS. WILL CLOSE. Use args like this: GrinGoldMinerCLI.exe configpath=C:\absolute\path\to\directory api-port=5777");
                }
                
                Console.ReadLine();
                Close();
            }
            #endregion

            try
            {
                var cudas = System.Diagnostics.Process.GetProcessesByName("CudaSolver");
                var ocls = System.Diagnostics.Process.GetProcessesByName("OclSolver");

                if (cudas.Count() > 0 || ocls.Count() > 0)
                {
                    Console.WriteLine("Existing CudaSolver or OclSolver processes running, please terminate them first.");
                    Console.ReadKey();
                }
            }
            catch { }

            GetConfig();

            #region init
            //ProMiner always in rolling mode
            Logger.consoleMode = ConsoleOutputMode.ROLLING_LOG;

            Logger.SetLogOptions(config.LogOptions);
            WorkerManager.Init(config);

            // this blocks on no connection ?
            ConnectionManager.Init(config, "grin29");

            #endregion

            //pro miner no TUI
            long counter = 0;
            while (!IsTerminated)
            {
                Task.Delay(1000).Wait();
                if (counter++ % 20 == 0)
                    WorkerManager.PrintWorkerInfo();
            }

            //while ended
            Close();
        }


        public static string WriteConfigToDisk(Config config)
        {
            Serialization.Serialize<Config>(config, configPath);
            return configPath;
        }

        private static void GetConfig()
        {
            var dir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            configPath = Path.Combine(Directory.GetParent(dir).FullName, "config.xml");

            config = new Config();
            if (File.Exists(configPath))
            {
                try
                {
                    config = Serialization.DeSerialize<Config>(configPath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Problem parsing config.xml, chcek that syntax and content is correct. (Ex: {ex.Message})");
                }
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
                    //Worker nv = new Worker(SharedSerialization.WorkerType.NVIDIA, 0, 0);
                    //var devices = nv.GetDevices();
                    var devices = WorkerManager.GetDevices(SharedSerialization.WorkerType.NVIDIA);

                    foreach (var dev in devices)
                    {
                        if (dev.memory > (long)1024 * 1024 * 1024 * 4)
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

                int nvcnt = generated_config.GPUOptions.Count;
                Console.WriteLine($"Detected {generated_config.GPUOptions.Count} suitable nvidia devices");

                Task.Delay(2000).Wait();

                // AMD
                try
                {
                    //Worker amd = new Worker(SharedSerialization.WorkerType.AMD, 0, 0);
                    //var devices = amd.GetDevices();
                    var devices = WorkerManager.GetDevices(SharedSerialization.WorkerType.AMD);

                    foreach (var dev in devices)
                    {
                        if ((dev.memory > (long)1024 * 1024 * 1024 * 3) && (dev.platformName.ToLower().Contains("amd") || dev.platformName.ToLower().Contains("advanced")))
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
                    Console.WriteLine($"Detected {generated_config.GPUOptions.Count - nvcnt} suitable AMD devices");
                }
                catch
                {
                    ;
                }

                if (generated_config.GPUOptions.Count == 0)
                    Console.WriteLine("No devices auto-detected, please use manual config (see readme)");

                string pool = "";
                string port = "13416";

                Console.WriteLine($"Enter your mining pool stratum address:");
                pool = Console.ReadLine();

                generated_config.PrimaryConnection.ConnectionAddress = pool;
                port = "13416";
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
                generated_config.PrimaryConnection.Ssl = ssl == "y" || ssl == "Y" || ssl == "YES" || ssl == "yes" || ssl == "是";
                Console.WriteLine($"Enter your email (pool login):");
                generated_config.PrimaryConnection.Login = Console.ReadLine();
                Console.WriteLine($"Enter your pool password:");
                generated_config.PrimaryConnection.Password = Console.ReadLine();


                WriteConfigToDisk(generated_config);
                Console.WriteLine();
                Console.WriteLine($"Created new config.xml in ({configPath}).");


                if (File.Exists(configPath))
                {
                    // try catch this !!
                    try
                    {
                        config = Serialization.DeSerialize<Config>(configPath);
                    }
                    catch
                    {
                        Console.WriteLine("Can't load generated config.");
                        Console.ReadLine();
                        Close();
                    }

                }
                else
                {
                    Console.WriteLine("Can't load generated config.");
                    Console.ReadLine();
                    Close();
                }
            }
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
                Console.CursorVisible = true;
                Environment.Exit(0);
            }
        }
    }
}
