using System;
using System.Collections.Generic;
using System.Text;
using SharedSerialization;

namespace CudaSolver
{
    public static class Logger
    {
        public static void Log(LogLevel level, string message, Exception e = null)
        {
            try
            {
                if (level != LogLevel.Debug)
                {
                    Console.WriteLine(string.Format("{0}:\t {1}, {2}, {3}", DateTime.Now, level.ToString(), message, e != null ? e.Message : ""));
                }
                lock (Comms.logsOut)
                {
                    Comms.logsOut.Enqueue(new LogMessage() { });
                }
                Comms.SetEvent();
            }
            catch
            {
                // log to file
            }
        }
    }




}
