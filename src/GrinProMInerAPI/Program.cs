//#define CHINA

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Mozkomor.GrinGoldMiner;

namespace GrinProMInerAPI
{
    public class Program
    {
        private static volatile bool IsTerminated;
        public static volatile bool ChangeRemoteTerminate = false;//API received new config
        public static Config config;
        static Dictionary<string, string> cmdParams = new Dictionary<string, string>();
        public static readonly bool IsLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        public const string ARG_CONFIGPATH = "configpath";
        public const string ARG_MODE = "mode"; //"rolling" for starting with rolling console mode
        public const string ARG_API_PORT = "api-port";
        public const string ARG_NO_CONFIG = "ignore-config";
        public const string ARG_STRATUM_ADDR = "stratum-address";
        public const string ARG_STRATUM_PORT = "stratum-port";
        public const string ARG_STRATUM_TLS = "stratum-tls";
        public const string ARG_STRATUM_LOGIN = "stratum-login";
        public const string ARG_STRATUM_PWD = "stratum-password";
        public const string ARG_STRATUM_BACKUP_ADDR = "stratum-backup-address";
        public const string ARG_STRATUM_BACKUP_PORT = "stratum-backup-port";
        public const string ARG_STRATUM_BACKUP_TLS = "stratum-backup-tls";
        public const string ARG_STRATUM_BACKUP_LOGIN = "stratum-backup-login";
        public const string ARG_STRATUM_BACKUP_PWD = "stratum-backup-password";
        public const string ARG_RECONNECT_TO_PRIMARY = "stratum-to-primary";
        public const string ARG_DASHBOARD = "dashboard"; //ip or ip:port
        public const string ARG_DASHBOARD_RIGNAME = "rig-name";
        public const string ARG_DADHBOARD_SENDINTERVAL = "dashboard-interval";
        public const string ARG_API_UTC = "api-utc"; //true/false, default true
        public const string ARG_AMD_DEVICES = "amd";
        public const string ARG_NVIDIA_DEVICES = "nvidia";
        public const string ARG_LOG_CONSOLE = "log-level-console";
        public const string ARG_LOG_FILE = "log-level-file";
        public const string ARG_LOG_DISABLE = "log-disable";

        private static string apiport = "5777";
        private static string configPath = "";


        public static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
 
            string algo = "grin29";

#if CHINA
            Console.WriteLine("starting GrinPro.io miner...");
            Console.WriteLine("运行GrinPro.io矿工");
#else
            Console.WriteLine("Starting GrinGoldMiner 3.0 API ...");
#endif

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
                PringArgsHelp();
                Console.ReadLine();
                Close();
            }
#endregion

#region start API
            if (cmdParams.ContainsKey(ARG_API_PORT))
                apiport = cmdParams[ARG_API_PORT];

            CreateWebHostBuilder(args).Build().Start(); //method defined below at end of file
#if CHINA
            Console.WriteLine("*******************************************************");
            Console.WriteLine($"HTML仪表盘位于http://localhost:{apiport}");
            Console.WriteLine($"JSON API在收听http://localhost:{apiport}/api");
            Console.WriteLine("         （两者也可通过网络获得）");
            Console.WriteLine("          (API文件：https://grinpro.io/api.html");
            Console.WriteLine("*******************************************************");
#else
            Console.WriteLine("*******************************************************");
            Console.WriteLine($"HTML Dashboard at http://localhost:{apiport}");
            Console.WriteLine($"JSON API listening on http://localhost:{apiport}/api");
            Console.WriteLine("         (Both also available over network)");
            Console.WriteLine("         (API docs: https://grinpro.io/api.html)");
            Console.WriteLine("*******************************************************");
#endif
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



            if (!HasArgValue(ARG_NO_CONFIG, "true"))
            {
                ///get config from file or generate config
                GetConfig();
            }
            else
            {
                bool HasAllNeeded = cmdParams.ContainsKey(ARG_STRATUM_ADDR)
                                    && cmdParams.ContainsKey(ARG_STRATUM_PORT)
                                    && cmdParams.ContainsKey(ARG_STRATUM_LOGIN)
                                    && cmdParams.ContainsKey(ARG_STRATUM_TLS)
                                    && cmdParams.ContainsKey(ARG_STRATUM_PORT)
                                    && (cmdParams.ContainsKey(ARG_AMD_DEVICES) || cmdParams.ContainsKey(ARG_NVIDIA_DEVICES));

                if (!HasAllNeeded)
                {
                    Console.WriteLine($"To run GrinPro.io {ARG_NO_CONFIG}=true, you MUST provide proper stratum connection details and GPU selectoin in arguments. Otherwise run without {ARG_NO_CONFIG} and use config.xml.");
                    PringArgsHelp();
                    Close();
                }

                try
                {
                    InMemoryConfigFromArgs();
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed parsing command like arguments needed for stratum connection and/or GPU options. " + ex.Message);
                    PringArgsHelp();
                    Close();
                }
            }

#region init
            //ProMiner always in rolling mode
            Logger.consoleMode = ConsoleOutputMode.ROLLING_LOG;

            Logger.SetLogOptions(config.LogOptions);
            WorkerManager.Init(config);

            // this blocks on no connection ?
            ConnectionManager.Init(config, algo);

            InitPingDashboard();

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

        private static void InMemoryConfigFromArgs()
        {
            Config cfg = Config.GetDefaultConfig();
            cfg.PrimaryConnection.ConnectionAddress = cmdParams[ARG_STRATUM_ADDR];
            cfg.PrimaryConnection.ConnectionPort = int.Parse(cmdParams[ARG_STRATUM_PORT]);
            cfg.PrimaryConnection.Login = cmdParams[ARG_STRATUM_LOGIN];
            cfg.PrimaryConnection.Password = (cmdParams.ContainsKey(ARG_STRATUM_PWD)) ? cmdParams[ARG_STRATUM_PWD] : "";
            cfg.PrimaryConnection.Ssl = bool.Parse(cmdParams[ARG_STRATUM_TLS]);


            bool HasBackupStratum = cmdParams.ContainsKey(ARG_STRATUM_BACKUP_ADDR)
                                    && cmdParams.ContainsKey(ARG_STRATUM_BACKUP_PORT)
                                    && cmdParams.ContainsKey(ARG_STRATUM_BACKUP_LOGIN)
                                    && cmdParams.ContainsKey(ARG_STRATUM_BACKUP_TLS);
            if (HasBackupStratum)
            {
                try
                {

                    cfg.SecondaryConnection.ConnectionAddress = cmdParams[ARG_STRATUM_BACKUP_ADDR];
                    cfg.SecondaryConnection.ConnectionPort = int.Parse(cmdParams[ARG_STRATUM_BACKUP_PORT]);
                    cfg.SecondaryConnection.Login = cmdParams[ARG_STRATUM_BACKUP_LOGIN];
                    cfg.SecondaryConnection.Password = (cmdParams.ContainsKey(ARG_STRATUM_BACKUP_PWD)) ? cmdParams[ARG_STRATUM_BACKUP_PWD] : "";
                    cfg.SecondaryConnection.Ssl = bool.Parse(cmdParams[ARG_STRATUM_BACKUP_TLS]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error parsing backup stratum connection details." + ex.Message);
                }
            }

            cfg.RemoteDashboard.DashboardAddress = TryGetArgValue(ARG_DASHBOARD);
            cfg.RemoteDashboard.RigName = TryGetArgValue(ARG_DASHBOARD_RIGNAME);
            if (cmdParams.ContainsKey(ARG_DADHBOARD_SENDINTERVAL))
                cfg.RemoteDashboard.SendInterval = int.Parse(TryGetArgValue(ARG_DADHBOARD_SENDINTERVAL));

            if (cmdParams.ContainsKey(ARG_RECONNECT_TO_PRIMARY))
            {
                try
                {
                    cfg.ReconnectToPrimary = int.Parse(cmdParams[ARG_RECONNECT_TO_PRIMARY]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error parsing " + ARG_RECONNECT_TO_PRIMARY + " " + ex.Message);
                }
            }

            if (cmdParams.ContainsKey(ARG_LOG_CONSOLE))
            {
                try { cfg.LogOptions.ConsoleMinimumLogLevel = (Mozkomor.GrinGoldMiner.LogLevel)(int.Parse(cmdParams[ARG_LOG_CONSOLE])); } catch { Console.WriteLine($"Problem parsing {ARG_LOG_CONSOLE}"); }
            }
            if (cmdParams.ContainsKey(ARG_LOG_FILE))
            {
                try { cfg.LogOptions.FileMinimumLogLevel = (Mozkomor.GrinGoldMiner.LogLevel)(int.Parse(cmdParams[ARG_LOG_FILE])); } catch { Console.WriteLine($"Problem parsing {ARG_LOG_FILE}"); }
            }
            if (cmdParams.ContainsKey(ARG_LOG_DISABLE))
            {
                try { cfg.LogOptions.DisableLogging = (bool.Parse(cmdParams[ARG_LOG_DISABLE])); } catch { Console.WriteLine($"Problem parsing {ARG_LOG_DISABLE}"); }
            }

            cfg.GPUOptions = new List<SharedSerialization.GPUOption>();

            if (cmdParams.ContainsKey(ARG_AMD_DEVICES))
            {
                var amd_devs = cmdParams[ARG_AMD_DEVICES].Split(',', StringSplitOptions.RemoveEmptyEntries); //0:2,0:3,0:4

                foreach (var pltf_id in amd_devs)
                {
                    var platform = pltf_id.Split(':')[0];
                    var id = pltf_id.Split(':')[1];
                    cfg.GPUOptions.Add(new SharedSerialization.GPUOption()
                    {
                        GPUName = $"AMD-{id}",
                        DeviceID = int.Parse(id),
                        Enabled = true,
                        GPUType = SharedSerialization.WorkerType.AMD,
                        PlatformID = int.Parse(platform)
                    });
                }
            }
            if (cmdParams.ContainsKey(ARG_NVIDIA_DEVICES))
            {
                var nv_devs = cmdParams[ARG_NVIDIA_DEVICES].Split(',', StringSplitOptions.RemoveEmptyEntries); //2,3,4

                foreach (var id in nv_devs)
                {
                    //add nvidia
                    cfg.GPUOptions.Add(new SharedSerialization.GPUOption()
                    {
                        GPUName = $"NVIDIA-{id}",
                        DeviceID = int.Parse(id),
                        Enabled = true,
                        GPUType = SharedSerialization.WorkerType.NVIDIA,
                        PlatformID = 0
                    });
                }
            }

            config = cfg;
        }

        public static string WriteConfigToDisk(Config config)
        {
            Serialization.Serialize<Config>(config, configPath);
            return configPath;
        }

        private static void GetConfig()
        {
            var dir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            if (cmdParams.ContainsKey(ARG_CONFIGPATH))
                dir = cmdParams[ARG_CONFIGPATH];
            configPath = Path.Combine(dir, "config.xml");
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

#if CHINA
                Console.WriteLine("No config file found, it will be generated now....");
                Console.WriteLine("无法找到配置文件，它即将立即生成…");
                Console.WriteLine("Autodetecting GPUs...");
                Console.WriteLine("自动检测图形处理器（GPUs)…");
#else
                Console.WriteLine("No config file found, it will be generated now....");
                Console.WriteLine("Autodetecting GPUs...");
#endif
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
#if CHINA
                Console.WriteLine($"Detected {generated_config.GPUOptions.Count} suitable nvidia devices");
                Console.WriteLine($"检测到{generated_config.GPUOptions.Count}个合适的英伟达（nvidia）装置");
#else
                Console.WriteLine($"Detected {generated_config.GPUOptions.Count} suitable nvidia devices");
#endif
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
#if CHINA
                    Console.WriteLine($"Detected {generated_config.GPUOptions.Count - nvcnt} suitable AMD devices");
                    Console.WriteLine($"检测到{generated_config.GPUOptions.Count - nvcnt}个合适的超微半导体（AMD）装置");
#else
                    Console.WriteLine($"Detected {generated_config.GPUOptions.Count - nvcnt} suitable AMD devices");
#endif
                }
                catch
                {
                    ;
                }

                if (generated_config.GPUOptions.Count == 0)
                    Console.WriteLine("No devices auto-detected, please use manual config (see readme)");

                string pool = "";
                string port = "13416";

#if CHINA
                    Console.WriteLine($"Enter your mining pool stratum address:");
                    Console.WriteLine($"输入您的矿池stratum地址：");
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
                        Console.WriteLine($"输入您的矿池stratum端口：");
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
                    Console.WriteLine($"使用传输层安全性协议（TLS)？（是/否） [注意矿池：端口必须支持TLS]");
                    string ssl = Console.ReadLine();
                    generated_config.PrimaryConnection.Ssl = ssl == "y" || ssl == "Y" || ssl == "YES" || ssl == "yes" || ssl == "是";
                    Console.WriteLine($"Enter your email (pool login):");
                    Console.WriteLine($"输入您的矿池登录名：");
                    generated_config.PrimaryConnection.Login = Console.ReadLine();
                    Console.WriteLine($"Enter your pool password:");
                    Console.WriteLine($"输入您的矿池密码：");
                    generated_config.PrimaryConnection.Password = Console.ReadLine();


                WriteConfigToDisk(generated_config);
                Console.WriteLine();
                Console.WriteLine($"Created new config.xml in ({configPath}).");
                Console.WriteLine($"在…建立一个新的config.xml ({configPath}).");
#else
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
#endif

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

        static void PringArgsHelp()
        {
            string help = @"Command line arguments for GrinPro.io miner.
General format is: arg1=value1 arg2=value2 etc.
Example:
Linux: ./GrinProMiner configpath=/absolute/path/to/directory api-port=3333
Windows: GrinProMiner.exe configpath=C:\absolute\path\to\directory api-port=3333

""configpath""		Defines directory where to save or load config.xml file from.
			Example:
			Linux: ./GrinProMiner configpath=/absolute/path/to/directory
			Windows: GrinProMiner.exe configpath=C:\absolute\path\to\directory
			More info about config.xml: https://github.com/mozkomor/GrinGoldMiner/blob/master/src/Master/config.xml
			
""api-port""		number of port on which JSON API should listen
			default value: 5777
            API docs: https://grinpro.io/api.html
			
Use following arguments ONLY if you can't use config.xml (because miner has no write rights etc)
This is not recommended, if you can, use config.xml it is preferred option.
Without config.xml you will lose some options like backup pool connection, CPU offload value, etc.
You can also upload/save config.xml via API: https://grinpro.io/api.html
More info about config.xml file: https://github.com/mozkomor/GrinGoldMiner/blob/master/src/Master/config.xml

In case you cannot use config.xml, you MUST use ignore-config=true argument and provide
all stratum arguments plus either amd or nvidia or both are required in this case.

""ignore-config""		""true"" if you want to skip loading config.xml, you MUST provide all stratum and GPU details as command line arguments
""stratum-address""	IP or dns address
			Example:
			stratum-address=eu.pool.com
""stratum-port""		stratum port number
""stratum-tls""		use encrypted connection, ""true"" or ""false"", pool must support on this port
""stratum-login""		stratum login
""stratum-password""	(optional) stratum password
""amd""			List of AMD cards for mining.
			Must be in this format OPENCL_PLATFORM:DEVICE_ID,OPENCL_PLATFORM:DEVICE_ID,OPENCL_PLATFORM:DEVICE_ID,..
			Example:
			amd=0:0,0:1,0:2 (will start mining on 3 AMD cards on OPENCL platform 0, cards have DEVICE IDs 0,1,2.
""nvidia""		List of NVIDIA cards for mining.
			Must be list of GPU DEVICE IDs separated by commas
			Example:
			nvidia=0,1,2,3 (will start mining on 4 NVIDIA cards with Device IDs 0,1,2,3
			nvidia=1 (will start mining on single NVIDIA card with Device ID 1)
""log-level-console""	(optional, default is 1) number 0/1/2/3, 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
""log-level-file""	(optional, default is 2) number 0/1/2/3, 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
""log-disable""		(optional) ""true"" will disable logging to file/disk. Otherwise log is written to logs/ directory

Example of config from arguments that will start mining on two GPUs - one Nvidia and one AMD, connected to grinmint pool:
Windows GrinProMiner.exe ignore-config=true stratum-address=eu-west-stratum.grinmint.com stratum-port=4416 stratum-tls=true stratum-login=login@example.com nvidia=0 amd=0:0
Linux: ./GrinProMiner ignore-config=true stratum-address=eu-west-stratum.grinmint.com stratum-port=4416 stratum-tls=true stratum-login=logina@example.com nvidia=0 amd=0:0 ";

            Console.WriteLine(help);
        }

        static void InitPingDashboard()
        {
            if (!string.IsNullOrEmpty(config?.RemoteDashboard?.DashboardAddress) && config?.RemoteDashboard?.SendInterval > 0)
                Task.Run(() => PingDashboard());
        }

        static void PingDashboard()
        {
            HttpClient client = new HttpClient();

            client.BaseAddress = config.RemoteDashboard.DashboardAddress.Contains(":")  //checked in calling method that not null
                ? new Uri($"http://{config.RemoteDashboard.DashboardAddress}")
                : new Uri($"http://{config.RemoteDashboard.DashboardAddress}:5888");

            var rigname = config.RemoteDashboard.RigName;
            var query_paramrigname = string.IsNullOrEmpty(rigname) ? "" : $"?name={rigname}";
            var query_apiport = (apiport != "5777") ? $"/{apiport}" : ""; //api port of this miner, not dashboard (for pulls from dasboard)
            var request_url = $"/api/hello{query_apiport}{query_paramrigname}";
            
            while (!IsTerminated && !ChangeRemoteTerminate) //TODO actualy whis will not work if the first run remoteDashboard was emptyand this task is notrunning...
            {
                Task.Delay(TimeSpan.FromSeconds(config.RemoteDashboard.SendInterval)).Wait();
                Console.WriteLine("http get: " + request_url);
                client.GetAsync(request_url);
            }

            if (ChangeRemoteTerminate)
            {
                //API send new config with potentially new remote dashboard setting
                //that set ChangeRemoteTerminate=true, which ended the while above
                //now we want to start pinging again and restart the bool indicator
                ChangeRemoteTerminate = false;
                InitPingDashboard();
            }

            client.Dispose();
        }

        static bool HasArgValue(string arg, string value) =>
            (cmdParams.ContainsKey(arg) && cmdParams[arg].Trim().ToLower() == value);

        static string TryGetArgValue(string arg) =>
            cmdParams.ContainsKey(arg) ? cmdParams[arg] : "";


        public static IWebHostBuilder CreateWebHostBuilder(string[] args) =>
            WebHost.CreateDefaultBuilder(args)
                .UseStartup<Startup>()
                .UseUrls($"http://0.0.0.0:{apiport}");
        //.UseUrls("http://localhost:5050", "http://0.0.0.0:5050");
    }
}
