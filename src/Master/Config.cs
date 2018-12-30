// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using System;
using System.Collections.Generic;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    public class Config
    {
        public string ConnectionAddress { get; set; }
        public int ConnectionPort { get; set; }
        public string Login { get; set; }
        public string Password { get; set; }

        public string SecondaryLogin { get; set; }
        public string SecondaryPassword { get; set; }
        public string SecondaryConnectionAddress { get; set; }
        public int SecondaryConnectionPort { get; set; }

        
        public static Config GetEmptyConfig()
        {
            return new Config() { ConnectionAddress = "", ConnectionPort = 0, Login = "", Password = "",
                SecondaryConnectionAddress  = "", SecondaryConnectionPort = 0, SecondaryLogin = "", SecondaryPassword = ""
                                };
        }
    }
}
