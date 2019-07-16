// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using SharedSerialization;
using System;
using System.Collections.Generic;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    public class Config
    {
        public Connection PrimaryConnection { get; set; }
        public Connection SecondaryConnection { get; set; }
        public int ReconnectToPrimary { get; set; }
        public DashboardOptions RemoteDashboard { get; set; }
        public LogOptions LogOptions { get; set; }
        public int CPUOffloadValue { get; set; }
        public List<GPUOption> GPUOptions { get; set; }


        public static Config GetDefaultConfig()
        {
            var c1 = new Connection() { ConnectionAddress = "us-east.stratum.grinmint.com", ConnectionPort = 13416, Login = "satoshi@nakamoto.com/rig21", Password = "myverystrongpassword", Ssl =false };
            var c2 = new Connection() { ConnectionAddress = "backup_pooladdress", ConnectionPort = 13416, Login = "login", Password = "password", Ssl = false };
            var logOptions = new LogOptions() {ConsoleMinimumLogLevel = LogLevel.INFO, FileMinimumLogLevel = LogLevel.WARNING, KeepDays=1, DisableLogging = false };
            var remoteDashboard = new DashboardOptions() { DashboardAddress = "", RigName = "", SendInterval=60 };
            List<GPUOption> gpuOptions = new List<GPUOption>() { new GPUOption() { DeviceID = 0, Enabled = true, GPUType = WorkerType.NVIDIA, PlatformID = 0 } };
            return new Config() { PrimaryConnection = c1,
                            SecondaryConnection = c2,
                            ReconnectToPrimary = 0,
                            RemoteDashboard = remoteDashboard,
                            CPUOffloadValue = 0,
                            GPUOptions = gpuOptions,
                            LogOptions = logOptions };
        }
    }

    public class DashboardOptions
    {
        public string DashboardAddress { get; set; }
        //public string DashboardPort { get; set; }
        public string RigName { get; set; }

        /// <summary>
        /// How often (in seconds) should miner push status info to remote dashboard
        /// If 0 then miner doesnt push, instead dashboard needs to pull (based on user interaction (refreshing page))
        /// </summary>
        public int SendInterval { get; set; }
    }

    public class Connection
    {
        public string ConnectionAddress { get; set; }
        public int ConnectionPort { get; set; }
        public bool Ssl { get; set; }
        public string Login { get; set; }
        public string Password { get; set; }
    }
}
