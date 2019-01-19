using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Mozkomor.GrinGoldMiner;

//Win 7? https://support.microsoft.com/en-us/help/2533623/microsoft-security-advisory-insecure-library-loading-could-allow-remot
namespace Mozkomor.GrinGoldMinerCLI
{
    class Program
    {
        private static volatile bool IsTerminated;
        public static Config config;
        static Dictionary<string, string> cmdParams = new Dictionary<string, string>();
        public static readonly bool IsLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        public const string ARG_CONFIGPATH = "configpath";
        public const string ARG_MODE = "mode"; //"rolling" for starting with rolling console mode

        static void Main(string[] args)
        {
            Console.CancelKeyPress += delegate
            {
                Console.WriteLine("Ctrl+C - Exitting");
                Close();
            };

            try
            {
                var cudas = System.Diagnostics.Process.GetProcessesByName("CudaSolver");
                var ocls = System.Diagnostics.Process.GetProcessesByName("OclSolver");

                if (cudas.Count() > 0 ||ocls.Count() > 0)
                {
                    Console.WriteLine("Existing CudaSolver or OclSolver processes running, please terminate them first.");
                    Console.ReadKey();
                }
            }
            catch { }


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
                    Console.WriteLine(@"ERROR PARSING ARGUMENTS. WILL CLOSE. Use args like this: ./GrinGoldMinerCLI configpath=/absolute/path/to/directory");
                }
                else
                {
                    Console.WriteLine(@"ERROR PARSING ARGUMENTS. WILL CLOSE. Use args like this: GrinGoldMinerCLI.exe configpath=C:\absolute\path\to\directory");
                }
                Console.ReadLine();
                Close();
            }

            //if (DateTime.Today >= new DateTime(2019, 1, 14))
            //{
            //    Console.WriteLine("!!! This version of GrinGoldMiner is outdated. Please go to https://github.com/mozkomor/GrinGoldMiner/releases and downlaod the latest release.");
            //    Logger.Log(LogLevel.ERROR, "!!! This version of GrinGoldMiner is outdated. Please go to https://github.com/mozkomor/GrinGoldMiner/releases and downlaod the latest release.");
            //    Console.ReadLine();
            //    Close();
            //}


            var dir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            if (cmdParams.ContainsKey(ARG_CONFIGPATH))
                dir = cmdParams[ARG_CONFIGPATH];
            var configPath = Path.Combine(dir, "config.xml");
            config = new Config();
            if (File.Exists(configPath))
            {
                try
                {
                    config = Serialization.DeSerialize<Config>(configPath);
                }
                catch(Exception ex)
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
                    Worker nv = new Worker(SharedSerialization.WorkerType.NVIDIA, 0, 0);
                    var devices = nv.GetDevices();

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
                    Worker amd = new Worker(SharedSerialization.WorkerType.AMD, 0, 0);
                    var devices = amd.GetDevices();

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
                Console.WriteLine($"Select minig pool (press number):");
                Console.WriteLine($"[1] Custom stratum address");
                Console.WriteLine($"[2] US-east grinmint.com");
                Console.WriteLine($"[3] EU-west grinmint.com");
                Console.WriteLine($"[4] mwgrinpool.com (open-source)");
                Console.WriteLine($"[5] EU grin-pool.org");
                Console.WriteLine("Or try some other pools (use option 1): cuckoomine.org grinpool.co sparkpool.com ");
                var key = Console.ReadLine();

                if (key == "2" || key == "3")
                {
                    generated_config.PrimaryConnection.ConnectionAddress = key == "2" ? "us-east-stratum.grinmint.com" : "eu-west-stratum.grinmint.com";
                    generated_config.PrimaryConnection.ConnectionPort = 4416;
                    generated_config.PrimaryConnection.Ssl = true;
                    Console.WriteLine($"Enter your email (pool login):");
                    var email = Console.ReadLine();
                   
                    if (email.Contains("/"))
                    {
                        generated_config.PrimaryConnection.Login = email;
                    }
                    else
                    {
                        Console.WriteLine($"Enter your rig name (e.g. rig1):");
                        var rig = Console.ReadLine();
                        if (string.IsNullOrWhiteSpace(rig)) rig = "rig1";
                        generated_config.PrimaryConnection.Login = $"{email}/{rig}";
                    }

                    Console.WriteLine($"Enter your pool password:");
                    generated_config.PrimaryConnection.Password = Console.ReadLine();
                }
                else if (key == "4")
                {
                    generated_config.PrimaryConnection.ConnectionAddress = "stratum.MWGrinPool.com";
                    generated_config.PrimaryConnection.ConnectionPort = 3334;
                    generated_config.PrimaryConnection.Ssl = true;
                    Console.WriteLine($"Enter your email (pool login):");
                    var email = Console.ReadLine();
                    generated_config.PrimaryConnection.Login = $"{email}";
                    Console.WriteLine($"Enter your pool password:");
                    generated_config.PrimaryConnection.Password = Console.ReadLine();
                }
                else if (key == "5")
                {
                    Console.WriteLine("You need to create account on grin-pool.org website first and then enter you login here.");
                    generated_config.PrimaryConnection.ConnectionAddress = "eu.stratum.grin-pool.org";
                    generated_config.PrimaryConnection.ConnectionPort = 3416;
                    generated_config.PrimaryConnection.Ssl = false;
                    Console.WriteLine($"Enter your login:");
                    var email = Console.ReadLine();
                    generated_config.PrimaryConnection.Login = $"{email}";
                    Console.WriteLine($"Enter your rig name (or leave empty):");
                    generated_config.PrimaryConnection.Password = Console.ReadLine();
                }
                else
                {

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
                    generated_config.PrimaryConnection.Ssl = ssl == "y" || ssl == "Y" || ssl == "YES" || ssl == "yes";
                    Console.WriteLine($"Enter your pool login:");
                    generated_config.PrimaryConnection.Login = Console.ReadLine();
                    Console.WriteLine($"Enter your pool password:");
                    generated_config.PrimaryConnection.Password = Console.ReadLine();
                }

                

                Serialization.Serialize<Config>(generated_config, configPath);
                if (generated_config.GPUOptions.Count > 2 && !IsLinux)
                {
                    Console.WriteLine();
                    Console.WriteLine($"You have {generated_config.GPUOptions.Count} GPUs, you MUST have at least { generated_config.GPUOptions.Count*8}GBs of Virtual Memory configured!");
                }
                Console.WriteLine();
                Console.WriteLine($"Created new config.xml in executable directory ({configPath}), edit this file to change miner settings.");
                Console.ReadLine();
                Close();
                return;
            }

            if (cmdParams.ContainsKey(ARG_MODE) && cmdParams[ARG_MODE] == "rolling")
                Logger.consoleMode = ConsoleOutputMode.ROLLING_LOG;

            Logger.SetLogOptions(config.LogOptions);

            Task.Factory.StartNew(() => { WriteGUI(); }, TaskCreationOptions.LongRunning);
            
            WorkerManager.Init(config);


            // this blocks on no connection ?
            ConnectionManager.Init(config);

            if (cmdParams.ContainsKey(ARG_MODE) && cmdParams[ARG_MODE] == "rolling")
            {
                while (!IsTerminated) Task.Delay(500).Wait();
            }
            else
            {
                bool checkKey = true;
                while (checkKey)
                {
                    if (Console.KeyAvailable)
                    {
                        ConsoleKeyInfo key = Console.ReadKey(true);
                        Console.Clear();
                        switch (key.Key)
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
                            case ConsoleKey.Q:
                                checkKey = false;
                                break;
                            case ConsoleKey.Enter:
                                Logger.criticalErrors.TryDequeue(out string msg);
                                break;
                            default:
                                break;
                        }
                    }
                    else
                        Task.Delay(100).Wait();
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
                Console.CursorVisible = true;
                Environment.Exit(0);
            }
        }

        private static long refreshes = 0;
        private static void WriteGUI()
        {
            Console.Clear();
            while (!IsTerminated)
            {
                refreshes++;

                switch (Logger.consoleMode)
                {
                    case ConsoleOutputMode.STATIC_TUI:
                        {
                            if (DateTime.Now.Second == 0) Console.Clear();
                            Console.CursorVisible = false;
                            Console.SetCursorPosition(0, 0);
                            WipeLine();

                            var conn = ConnectionManager.GetCurrConn();

                            string remote = "---";
                            string status = "CONNECTING";

                            if (conn != null)
                            {
                                remote = $"{conn.ip}:{conn.port}";
                                status = conn.IsConnected ? "CONNECTED" : "DISCONNECTED";
                                
                            }
                            string who = ConnectionManager.IsInFee() ? "FEE" : (conn?.login ?? "USER");
                            who = who.Length > 10 ? who.Substring(0, 10)+".." : who;

                            Console.WriteLine("Grin Gold Miner 2.8 - Dreaming in cycles");
                            Console.WriteLine("------------------------------------------------------------------------------------------");
                            WipeLine();
                            Console.Write("Mining for: ");
                            Console.CursorLeft = 22;
                            Console.ForegroundColor = who == "FEE" ? ConsoleColor.Blue : ConsoleColor.Cyan;
                            Console.Write(who); Console.ResetColor(); //TODO
                            Console.CursorLeft = 35;
                            Console.WriteLine($"Stratum Server: {remote}"); //TODO
                            WipeLine();
                            Console.Write("Connection status: ");
                            Console.CursorLeft = 22;
                            Console.ForegroundColor = status == "CONNECTED" ? ConsoleColor.Green : (status == "CONNECTING" ? ConsoleColor.Yellow : ConsoleColor.Red);
                            Console.Write(status); Console.ResetColor(); //TODO

                            Console.CursorLeft = 35;
                            Console.WriteLine($"Last job in:    {(DateTime.Now-WorkerManager.lastJob).TotalSeconds:F0} seconds");
                            WipeLine();
                            Console.Write("Shares (sub/acc/rej): ");
                            Console.CursorLeft = 22;
                            if (conn != null)
                            {
                                Console.ForegroundColor = ConsoleColor.Yellow;
                                Console.Write($"{conn.totalShares}"); Console.ResetColor(); Console.Write("/");
                                if (conn.sharesAccepted > 0) Console.ForegroundColor = ConsoleColor.Green;
                                Console.Write($"{conn.sharesAccepted}"); Console.ResetColor(); Console.Write("/");
                                if (conn.sharesRejected + conn.sharesTooLate > 0) Console.ForegroundColor = ConsoleColor.Red;
                                Console.Write($"{conn.sharesRejected + conn.sharesTooLate}"); Console.ResetColor();
                                Console.ResetColor();


                                Console.CursorLeft = 35;
                                Console.WriteLine($"Last share:     {(DateTime.Now - conn.lastShare).TotalSeconds:F0} seconds");
                                WipeLine();
                            }
                            else
                                Console.WriteLine();
                            if (who == "FEE")
                            {
                                Console.ForegroundColor = ConsoleColor.DarkMagenta;
                                Console.WriteLine("GGM collects 1% as fee for the Grin Development Fund and 1% for further miner development.");
                                Console.WriteLine("Thank you very much for your support. It makes a difference.");
                                //ConnectionManager.printHeart();
                                Console.ResetColor();
                                WipeLine();
                            }
                            Console.WriteLine("------------------------------------------------------------------------------------------");
                            WipeLine();
                            foreach (var w in WorkerManager.workers)
                            {
                                w.PrintStatusLinesToConsole();
                            }
                            Console.WriteLine("------------------------------------------------------------------------------------------");
                            WipeLines(9);
                            if (Logger.criticalErrors.TryPeek(out string msg))
                            {
                                if (msg.Contains("GPU") || msg.ToLower().Contains("login"))
                                {
                                    Console.ForegroundColor = ConsoleColor.Red;
                                    Console.WriteLine($"ERROR [hit Enter to hide]: {msg}");
                                    Console.ResetColor();
                                    Console.WriteLine("------------------------------------------------------------------------------------------");
                                }
                                else
                                    Logger.criticalErrors.TryDequeue(out string msgout);
                            }
                            //Console.ForegroundColor = ConsoleColor.Yellow;
                            //Console.WriteLine("Last log messages:"); Console.ResetColor();
                            WipeLines(5);
                            Console.WriteLine(Logger.GetlastLogs());
                            WipeLine();
                        }
                        break;
                }

                Task.Delay(500).Wait();

                if (refreshes % 16 == 0)
                {
                    try
                    {
                        // debug dump
                        var conn = ConnectionManager.GetCurrConn();
                        if (conn != null)
                            Logger.Log(LogLevel.DEBUG, $"Statistics for {conn.id}: shares sub: {conn.totalShares} ac: {conn.sharesAccepted} rj: {conn.sharesRejected + conn.sharesTooLate}");

                        foreach (var w in WorkerManager.workers)
                        {
                            w.PrintStatusLinesToLog();
                        }
                    }
                    catch { }
                }
            }
        }

        private static void WipeLine()
        {
            Console.Write("                                                                                                     ");
            Console.CursorLeft = 0;
        }
        private static void WipeLines(int cnt)
        {
            for (int i = 0; i < cnt; i++)
            {
                Console.WriteLine("                                                                                                       ");
                Console.CursorLeft = 0;
            }
            Console.CursorTop -= cnt;
        }
    }
}
