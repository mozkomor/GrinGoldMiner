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
        public LogOptions LogOptions { get; set; }
        public int CPUOffloadValue { get; set; }
        public List<GPUOption> GPUOptions { get; set; }


        public static Config GetDefaultConfig()
        {
            var c1 = new Connection() { ConnectionAddress = "us-east.stratum.grinmint.com", ConnectionPort = 13416, Login = "satoshi@nakamoto.com/rig21", Password = "myverystrongpassword", Ssl =false };
            var c2 = new Connection() { ConnectionAddress = "backup_pooladdress", ConnectionPort = 13416, Login = "login", Password = "password", Ssl = false };
            var logOptions = new LogOptions() {ConsoleMinimumLogLevel = LogLevel.INFO, FileMinimumLogLevel = LogLevel.WARNING, KeepDays=1, DisableLogging = false };
            List<GPUOption> gpuOptions = new List<GPUOption>() { new GPUOption() { DeviceID = 0, Enabled = true, GPUType = WorkerType.NVIDIA, PlatformID = 0 } };
            return new Config() { PrimaryConnection = c1, SecondaryConnection = c2, CPUOffloadValue = 0, GPUOptions = gpuOptions, LogOptions = logOptions };
        }
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
