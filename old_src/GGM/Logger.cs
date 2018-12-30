// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.Text;

namespace GGM
{
    class Logger
    {
        internal static void Log(LogType type, string s, Exception ex = null)
        {

            Console.WriteLine(type.ToString() + ": " + s + (ex == null ? "" : ex.Message));

        }

        internal static void PushError(string v)
        {
            
        }
    }

    public enum LogType
    {
        FatalError,
        Error,
        Info,
        Debug,
        Network,
        Solutions
    }

}
